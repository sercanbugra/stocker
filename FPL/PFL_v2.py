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
all_history["prev_points"] = all_history.groupby("Player")["total_points"].shift(1)
all_history["rolling_mean_3"] = all_history.groupby("Player")["total_points"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
all_history["rolling_std_3"] = all_history.groupby("Player")["total_points"].rolling(3, min_periods=1).std().reset_index(level=0, drop=True)
all_history["value_change"] = all_history.groupby("Player")["value"].diff(1)

all_history = all_history.dropna(subset=["prev_points", "rolling_mean_3"]).reset_index(drop=True)
print(f"📊 After filtering: {all_history['Player'].nunique()} players, {len(all_history)} valid rows.")

# === 5. Prepare training/validation ===
features = ["prev_points", "rolling_mean_3", "rolling_std_3", "value", "value_change"]
target = "total_points"

X = all_history[features]
y = all_history[target]
train_mask = all_history["round"] - 1 < all_history["round"].max() - 2
X_train, X_valid = X[train_mask], X[~train_mask]
y_train, y_valid = y[train_mask], y[~train_mask]

print(f"🧠 Training rows: {len(X_train)}, Validation rows: {len(X_valid)}")

# === 6. Train LightGBM ===
print("\n🚀 Training LightGBM...")
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=["train", "valid"],
    num_boost_round=5000,
    callbacks=[early_stopping(100), log_evaluation(100)],
)

pred_valid_lgb = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration)
rmse_lgb = sqrt(mean_squared_error(y_valid, pred_valid_lgb))
print(f"✅ LightGBM RMSE: {rmse_lgb:.4f}")

# === 7. Train XGBoost ===
print("\n🚀 Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.05,
    n_estimators=5000,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=100,
    eval_metric="rmse",
    verbosity=1,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)

pred_valid_xgb = xgb_model.predict(X_valid)
rmse_xgb = sqrt(mean_squared_error(y_valid, pred_valid_xgb))
print(f"✅ XGBoost RMSE: {rmse_xgb:.4f}")

# === 8. Forecast next GW ===
print("\n📈 Forecasting next GW points...")

latest_round = all_history["round"].max()
latest_data = (
    all_history.sort_values(["Player", "round"])
    .groupby("Player")
    .tail(1)
    .copy()
)

X_next = latest_data[features]
latest_data["Pred_LightGBM"] = lgb_model.predict(X_next, num_iteration=lgb_model.best_iteration)
latest_data["Pred_XGBoost"] = xgb_model.predict(X_next)
latest_data["Predicted_Avg"] = (latest_data["Pred_LightGBM"] + latest_data["Pred_XGBoost"]) / 2

# === 9. Pivot all weeks into columns ===
print("\n📊 Pivoting weekly scores...")

pivot_scores = (
    all_history.pivot_table(index=["Player", "Team", "Position"],
                            columns="round",
                            values="total_points")
    .add_prefix("W")
    .reset_index()
)

# Merge forecasts
final_output = pivot_scores.merge(
    latest_data[["Player", "Pred_LightGBM", "Pred_XGBoost", "Predicted_Avg"]],
    on="Player",
    how="left"
)

# === 10. Save all to Excel ===
output_file = "fpl_forecast_all_weeks.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    final_output.to_excel(writer, index=False, sheet_name="Forecasts")
    summary_df = pd.DataFrame({
        "Model": ["LightGBM", "XGBoost"],
        "Validation_RMSE": [rmse_lgb, rmse_xgb],
    })
    summary_df.to_excel(writer, index=False, sheet_name="Model_Summary")

print(f"\n💾 Excel file created: '{output_file}'")
print(f"🧮 Total players: {len(final_output)}")
