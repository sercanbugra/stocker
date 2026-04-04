import pandas as pd
import requests


def get_fpl_data():
    """
    Fetch and process player statistics from the official FPL API.
    Includes appearances, performance, and discipline data.
    """

    base_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    resp = requests.get(base_url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    elements = data["elements"]
    teams = {team["id"]: team["name"] for team in data["teams"]}
    positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    df = pd.DataFrame(elements)

    # --- Derived Columns ---
    df["Team"] = df["team"].map(teams)
    df["Position"] = df["element_type"].map(positions)
    df["Last Week Price (£m)"] = df["now_cost"] / 10.0

    # Calculate approximate appearances (based on minutes played)
    df["Appearances"] = (df["minutes"] / 90).round(0).astype(int)

    # Median & Weighted averages for points
    df["Median Points"] = df["points_per_game"].astype(float)
    df["Weighted Avg Points"] = (
        df["form"].astype(float) * 0.7 + df["points_per_game"].astype(float) * 0.3
    )

    # Goals / Assists per appearance
    df["Goals per Appearance"] = df.apply(
        lambda x: x["goals_scored"] / x["Appearances"] if x["Appearances"] > 0 else 0,
        axis=1,
    )
    df["Assists per Appearance"] = df.apply(
        lambda x: x["assists"] / x["Appearances"] if x["Appearances"] > 0 else 0,
        axis=1,
    )

    # Discipline Index (yellow + 3×red)
    df["Discipline Index"] = df["yellow_cards"] + 3 * df["red_cards"]

    # Chance of playing next week
    df["Chance of Playing Next Week"] = df["chance_of_playing_next_round"].fillna(0).apply(
        lambda x: f"{int(x)}%" if x > 0 else "Unavailable"
    )

    # --- Build Output DataFrame ---
    df_out = pd.DataFrame({
        "Player": df["web_name"],
        "Position": df["Position"],
        "Team": df["Team"],
        "Last Week Price (£m)": df["Last Week Price (£m)"].round(1),
        "Median Points": df["Median Points"].round(2),
        "Weighted Avg Points": df["Weighted Avg Points"].round(2),
        "Appearances": df["Appearances"],
        "Goals": df["goals_scored"],
        "Assists": df["assists"],
        "Goals per Appearance": df["Goals per Appearance"].round(2),
        "Assists per Appearance": df["Assists per Appearance"].round(2),
        "Yellow Cards": df["yellow_cards"],
        "Red Cards": df["red_cards"],
        "Discipline Index": df["Discipline Index"],
        "Total Points": df["total_points"].astype(int),
    })

    # Sort by Total Points
    df_out.sort_values(by="Total Points", ascending=False, inplace=True)

    return df_out
