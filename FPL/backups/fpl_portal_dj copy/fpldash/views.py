import os
import requests
from django.http import JsonResponse
from django.shortcuts import render
from .fpl_data import get_fpl_data

TEAM_ID = os.getenv("FPL_TEAM_ID", "1897520")

def _get_last_gameweek_points(player_id: int) -> int:
    try:
        url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return 0
        data = resp.json()
        if "history" in data and data["history"]:
            return data["history"][-1].get("total_points", 0)
    except Exception:
        pass
    return 0

def _get_fpl_team(manager_id: str):
    try:
        base = "https://fantasy.premierleague.com/api/"
        bootstrap = requests.get(f"{base}bootstrap-static/", timeout=15).json()
        elements = bootstrap["elements"]
        player_map = {p["id"]: p for p in elements}
        teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
        positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

        current_gw = next((gw["id"] for gw in bootstrap["events"] if gw["is_current"]), None)
        if not current_gw:  # güvenlik
            current_gw = max([gw["id"] for gw in bootstrap["events"] if gw["is_next"] or gw["finished"]], default=1)

        picks = requests.get(f"{base}entry/{manager_id}/event/{current_gw}/picks/", timeout=15).json()
        if "picks" not in picks:
            return []

        out = []
        for pick in picks["picks"]:
            player = player_map.get(pick["element"])
            if not player:
                continue
            photo_id = str(player["photo"]).split(".")[0]
            photo = f"https://resources.premierleague.com/premierleague/photos/players/250x250/p{photo_id}.png"

            out.append({
                "id": player["id"],
                "web_name": player["web_name"],
                "name": f"{player['first_name']} {player['second_name']}",
                "team": teams.get(player["team"], "Unknown"),
                "position": positions.get(player["element_type"], "N/A"),
                "now_cost": round(player["now_cost"] / 10.0, 1),
                "points": int(player["total_points"]),
                "last_gw_points": _get_last_gameweek_points(player["id"]),
                "is_captain": pick.get("is_captain", False),
                "is_vice_captain": pick.get("is_vice_captain", False),
                "photo": photo,
                "starting": pick["position"] <= 11,
            })
        out.sort(key=lambda x: (not x["starting"], -x["last_gw_points"]))
        return out
    except Exception:
        return []

# ---------- Views ----------
def index(request):
    return render(request, "fpldash/index.html")

def api_myteam(request):
    return JsonResponse(_get_fpl_team(TEAM_ID), safe=False)

def api_data(request):
    df = get_fpl_data()
    return JsonResponse(df.to_dict(orient="records"), safe=False)

def api_suggestions(request):
    # Basit sabit öneriler; sonra akıllandırılabilir
    suggestions = [
        {"name": "Cole Palmer", "team": "Chelsea", "position": "MID", "reason": "In-form & good fixtures"},
        {"name": "Anthony Gordon", "team": "Newcastle", "position": "MID", "reason": "Consistent returns"},
        {"name": "Evan Ferguson", "team": "Brighton", "position": "FWD", "reason": "Budget forward"},
    ]
    return JsonResponse(suggestions, safe=False)
