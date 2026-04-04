import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt
from lightgbm import early_stopping, log_evaluation

# === 1. Fetch base info ===
base_url = "https://fantasy.premierleague.com/api/"
bootstrap = requests.get(base_url + "bootstrap-static/").json()

players = pd.DataFrame(bootstrap["elements"])
teams = pd.DataFrame(bootstrap["teams"])[["id", "name"]]
team_map = dict(zip(teams["id"], teams["name"]))
positions = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}

players["position"] = players["element_type"].map(positions)
players["team_name"] = players["team"].map(team_map)

# === 2. Collect player history ===
data_list = []
print("\n📥 Fetching player histories...")

for _, row in tqdm(players.iterrows(), total=len(players)):
    player_id = row["id"]
    player_name = f"{row['first_name']} {row['second_name']}"
    position = row["position"]
    team = row["team_name"]

    try:
        summary = requests.get(f"{base_url}element-summary/{player_id}/").json()
        history = pd.DataFrame(summary["history"])
        if history.empty:
            continue

        history = history[["round", "total_points", "value"]]
        history["value"] = history["value"] / 10.0
        history["Player"] = player_name
        history["Position"] = position
        history["Team"] = team

        data_list.append(history)
    except Exception:
        continue

# === 3. Combine ===
all_history = pd.concat(data_list, ignore_index=True)
print(f"\n✅ Collected data for {all_history['Player'].nunique()} players ({len(all_history)} total rows).")

# === 4. Feature engineering ===
print("🧩 Creating lag/rolling features...")

all_history = all_history.sort_values(["Player", "round"]).reset_index(drop=True)

latest_round = all_history["round"].max()
train_history = all_history[all_history["round"] < latest_round].copy()  # exclude last week

train_history["prev_points"] = train_history.groupby("Player")["total_points"].shift(1)
train_history["rolling_mean_3"] = train_history.groupby("Player")["total_points"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
train_history["rolling_std_3"] = train_history.groupby("Player")["total_points"].rolling(3, min_periods=1).std().reset_index(level=0, drop=True)
train_history["value_change"] = train_history.groupby("Player")["value"].diff(1)

train_history = train_history.dropna(subset=["prev_points", "rolling_mean_3"]).reset_index(drop=True)
print(f"📊 After filtering: {train_history['Player'].nunique()} players, {len(train_history)} valid rows.")

# === 5. Prepare training ===
features = ["prev_points", "rolling_mean_3", "rolling_std_3", "value", "value_change"]
target = "total_points"

X_train = train_history[features]
y_train = train_history[target]

# === 6. Prepare next GW features safely ===
latest_data = all_history.groupby("Player").tail(1).copy()  # keep last week info for Excel

# Calculate prev_points, rolling_mean_3, rolling_std_3, value_change using only weeks 1 to N-1
prev_points = all_history[all_history["round"] == latest_round - 1][["Player", "total_points"]].rename(columns={"total_points":"prev_points"})
rolling_mean_3 = (
    all_history[all_history["round"] < latest_round].groupby("Player")["total_points"]
    .apply(lambda x: x.tail(3).mean())
    .reset_index()
    .rename(columns={"total_points":"rolling_mean_3"})
)
rolling_std_3 = (
    all_history[all_history["round"] < latest_round].groupby("Player")["total_points"]
    .apply(lambda x: x.tail(3).std())
    .reset_index()
    .rename(columns={"total_points":"rolling_std_3"})
)
val_prev = all_history[all_history["round"] == latest_round - 1][["Player", "value"]].rename(columns={"value":"value_prev"})
val_before = all_history[all_history["round"] == latest_round - 2][["Player", "value"]].rename(columns={"value":"value_before"})
val_change = pd.merge(val_prev, val_before, on="Player", how="left")
val_change["value_change"] = val_change["value_prev"] - val_change["value_before"]

# Merge all features
latest_data = latest_data.merge(prev_points, on="Player", how="left")
latest_data = latest_data.merge(rolling_mean_3, on="Player", how="left")
latest_data = latest_data.merge(rolling_std_3, on="Player", how="left")
latest_data = latest_data.merge(val_change[["Player","value_change"]], on="Player", how="left")

X_next = latest_data[features]

# === 7. Train LightGBM ===
print("\n🚀 Training LightGBM...")
lgb_train = lgb.Dataset(X_train, y_train)
lgb_model = lgb.train(
    {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    },
    lgb_train,
    num_boost_round=1000
)
pred_next_lgb = lgb_model.predict(X_next)

# === 8. Train XGBoost ===
print("\n🚀 Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=1,
)
xgb_model.fit(X_train, y_train)
pred_next_xgb = xgb_model.predict(X_next)

# Combine predictions
latest_data["Pred_LightGBM"] = pred_next_lgb
latest_data["Pred_XGBoost"] = pred_next_xgb
latest_data["Predicted_Avg"] = (latest_data["Pred_LightGBM"] + latest_data["Pred_XGBoost"]) / 2

# === 9. Pivot all weeks into columns (keep last week) ===
print("\n📊 Pivoting weekly scores (all weeks included)...")
pivot_scores = (
    all_history.pivot_table(
        index=["Player", "Team", "Position"],
        columns="round",
        values="total_points"
    )
    .add_prefix("W")
    .reset_index()
)

# Merge predictions
final_output = pivot_scores.merge(
    latest_data[["Player", "Pred_LightGBM", "Pred_XGBoost", "Predicted_Avg"]],
    on="Player",
    how="left"
)

# === 10. Save to Excel ===
output_file = "fpl_forecast_next_gw.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    final_output.to_excel(writer, index=False, sheet_name="Forecasts")

print(f"\n💾 Excel file created: '{output_file}'")
print(f"🧮 Total players: {len(final_output)}")
