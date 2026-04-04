# fpl_data.py
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

BASE_URL = "https://fantasy.premierleague.com/api/"

def get_fpl_data():
    # bootstrap-static has all player meta
    bootstrap = requests.get(BASE_URL + "bootstrap-static/").json()
    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])

    # Position & team mapping
    positions = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}
    players["position"] = players["element_type"].map(positions)
    team_map = teams.set_index("id")["name"].to_dict()
    players["team_name"] = players["team"].map(team_map)

    # Top 20 per position by total_points
    top_players = (
        players.groupby("position", group_keys=False)
        .apply(lambda x: x.sort_values("total_points", ascending=False).head(20))
        [["id", "first_name", "second_name", "team_name", "position", "total_points", "chance_of_playing_next_round"]]
    ).reset_index(drop=True)

    rows = []
    for _, row in tqdm(top_players.iterrows(), total=len(top_players)):
        pid = row["id"]
        name = f"{row['first_name']} {row['second_name']}"
        pos = row["position"]
        team = row["team_name"]
        chance = row["chance_of_playing_next_round"]

        try:
            summary = requests.get(f"{BASE_URL}element-summary/{pid}/").json()
            history = pd.DataFrame(summary["history"])
        except Exception:
            continue

        if history.empty:
            continue

        # price is in tenths of £m
        history["value"] = history["value"] / 10.0
        last_week_price = float(history["value"].iloc[-1])

        median_points = float(history["total_points"].median())
        weights = np.linspace(0.5, 1.0, len(history))
        weighted_avg = float(np.average(history["total_points"], weights=weights))

        appearances = int(len(history))
        goals = int(history["goals_scored"].sum())
        assists = int(history["assists"].sum())
        yellow_cards = int(history["yellow_cards"].sum())
        red_cards = int(history["red_cards"].sum())

        goals_per_app = round(goals / appearances, 2) if appearances > 0 else 0.0
        assists_per_app = round(assists / appearances, 2) if appearances > 0 else 0.0
        discipline_index = int(yellow_cards + 2 * red_cards)

        if pd.isna(chance):
            playable_status = "Unknown"
        elif chance >= 75:
            playable_status = "Likely to Play"
        elif chance > 0:
            playable_status = "Doubtful"
        else:
            playable_status = "Out"

        rows.append({
            "Player": name,
            "Position": pos,
            "Team": team,
            "Last Week Price (£m)": round(last_week_price, 1),
            "Median Points": round(median_points, 2),
            "Weighted Avg Points": round(weighted_avg, 2),
            "Chance of Playing Next Week": playable_status,
            "Appearances": appearances,
            "Goals": goals,
            "Assists": assists,
            "Goals per Appearance": goals_per_app,
            "Assists per Appearance": assists_per_app,
            "Yellow Cards": yellow_cards,
            "Red Cards": red_cards,
            "Discipline Index": discipline_index,
            "Total Points": int(row["total_points"])
        })

    df = pd.DataFrame(rows)
    # Ensure Player + Position first
    ordered = ["Player", "Position"] + [c for c in df.columns if c not in ["Player", "Position"]]
    return df[ordered]
