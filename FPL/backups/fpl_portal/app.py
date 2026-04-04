from flask import Flask, render_template, jsonify
import os
import requests
from dotenv import load_dotenv
from fpl_data import get_fpl_data

# ===================================
# INITIAL SETUP
# ===================================
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super_secret_key")

TEAM_ID = os.getenv("FPL_TEAM_ID", "1897520")


# ===================================
# HELPER FUNCTIONS
# ===================================
def get_last_gameweek_points(player_id):
    """Fetch last gameweek's points for a player"""
    try:
        url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(f"⚠️ Failed to fetch summary for player {player_id}")
            return 0
        data = resp.json()
        if "history" in data and len(data["history"]) > 0:
            return data["history"][-1].get("total_points", 0)
    except Exception as e:
        print(f"⚠️ Error fetching last GW points: {e}")
    return 0


def get_fpl_team(manager_id):
    """Fetch FPL manager's team"""
    try:
        base_url = "https://fantasy.premierleague.com/api/"
        bootstrap = requests.get(f"{base_url}bootstrap-static/").json()

        elements = bootstrap["elements"]
        player_map = {p["id"]: p for p in elements}
        teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
        positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

        current_gw = next((gw["id"] for gw in bootstrap["events"] if gw["is_current"]), None)
        picks_url = f"{base_url}entry/{manager_id}/event/{current_gw}/picks/"
        picks_data = requests.get(picks_url).json()

        if "picks" not in picks_data:
            return []

        players = []
        for pick in picks_data["picks"]:
            player = player_map.get(pick["element"])
            if not player:
                continue
            photo_id = str(player["photo"]).split(".")[0]
            img = f"https://resources.premierleague.com/premierleague/photos/players/250x250/p{photo_id}.png"

            players.append({
                "id": player["id"],
                "web_name": player["web_name"],
                "name": f"{player['first_name']} {player['second_name']}",
                "team": teams[player["team"]],
                "position": positions[player["element_type"]],
                "now_cost": round(player["now_cost"] / 10.0, 1),
                "points": int(player["total_points"]),
                "last_gw_points": get_last_gameweek_points(player["id"]),
                "is_captain": pick.get("is_captain", False),
                "is_vice_captain": pick.get("is_vice_captain", False),
                "photo": img,
                "starting": pick["position"] <= 11
            })
        players.sort(key=lambda x: (not x["starting"], -x["last_gw_points"]))
        return players

    except Exception as e:
        print(f"❌ Error fetching team: {e}")
        return []


# ===================================
# ROUTES
# ===================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/myteam")
def api_myteam():
    return jsonify(get_fpl_team(TEAM_ID))


@app.route("/api/data")
def api_data():
    df = get_fpl_data()
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/suggestions")
def api_suggestions():
    """Mock player suggestions"""
    suggestions = [
        {"name": "Cole Palmer", "team": "Chelsea", "position": "MID", "reason": "Good fixtures"},
        {"name": "Anthony Gordon", "team": "Newcastle", "position": "MID", "reason": "Strong home record"},
        {"name": "Evan Ferguson", "team": "Brighton", "position": "FWD", "reason": "Budget striker"},
    ]
    return jsonify(suggestions)


if __name__ == "__main__":
    print("🚀 Starting FPL Dashboard → http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
