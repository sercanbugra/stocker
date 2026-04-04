import requests
import pandas as pd
from tqdm import tqdm

# === 1. Base URLs ===
base_url = "https://fantasy.premierleague.com/api/"

# === 2. Pull global data (all players) ===
bootstrap = requests.get(base_url + "bootstrap-static/").json()
players = pd.DataFrame(bootstrap["elements"])
positions = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}
players["position"] = players["element_type"].map(positions)

# === 3. Top 10 players per position by total points ===
top_players = (
    players.groupby("position", group_keys=False)
    .apply(lambda x: x.sort_values("total_points", ascending=False).head(20))
    [["id", "first_name", "second_name", "position", "team", "total_points"]]
).reset_index(drop=True)

print("📊 Top-scoring players:")
print(top_players.head())

# === 4. Fetch weekly scores + last week price for each top player ===
data_list = []

for _, row in tqdm(top_players.iterrows(), total=len(top_players)):
    player_id = row["id"]
    player_name = f"{row['first_name']} {row['second_name']}"
    position = row["position"]

    summary = requests.get(f"{base_url}element-summary/{player_id}/").json()
    history = pd.DataFrame(summary["history"])

    if not history.empty:
        # Weekly points
        history = history[["round", "total_points", "value"]]
        history["Player"] = player_name
        history["Position"] = position

        # Convert price (e.g. 125 -> 12.5)
        history["value"] = history["value"] / 10.0

        # Capture last week’s price
        last_week_price = history["value"].iloc[-1]

        # Add to dataset
        data_list.append((history, last_week_price))

# === 5. Combine all weekly scores ===
all_history = pd.concat([h[0] for h in data_list])
pivot = all_history.pivot_table(
    index=["Position", "Player"],
    columns="round",
    values="total_points",
    fill_value=0
).reset_index()

pivot.columns = ["Position", "Player"] + [f"GW{c}" for c in pivot.columns[2:]]
pivot["Total Points"] = pivot[[c for c in pivot.columns if c.startswith("GW")]].sum(axis=1)

# === 6. Add last week price column ===
price_map = {h[0]["Player"].iloc[0]: h[1] for h in data_list}
pivot["Last Week Price (£m)"] = pivot["Player"].map(price_map)

# === 7. Sort & export ===
pivot = pivot.sort_values(["Position", "Total Points"], ascending=[True, False])
pivot.to_csv("fpl_top20_by_position_with_price_2024_25.csv", index=False)

print("\n✅ Table created successfully!")
print(pivot.head(20))
print("\n💾 Saved as 'fpl_top20_by_position_with_price_2024_25.csv'")
