import requests

def get_fpl_team(manager_id):
    """
    Fetches and displays an FPL manager's team for the current gameweek.
    """
    try:
        # 1. Get general FPL data (players, teams, gameweeks)
        base_url = "https://fantasy.premierleague.com/api/"
        bootstrap_data = requests.get(f"{base_url}bootstrap-static/").json()

        # 2. Create a mapping of player IDs to player names
        elements = bootstrap_data['elements']
        player_map = {player['id']: player['web_name'] for player in elements}

        # 3. Find the current gameweek
        events = bootstrap_data['events']
        current_gameweek = None
        for gw in events:
            if gw['is_current']:
                current_gameweek = gw['id']
                break
        
        if not current_gameweek:
            print("Could not determine the current gameweek.")
            return

        print(f"Fetching team for Gameweek {current_gameweek}...\n")

        # 4. Get the manager's team picks for the current gameweek
        picks_url = f"{base_url}entry/{manager_id}/event/{current_gameweek}/picks/"
        picks_data = requests.get(picks_url).json()
        
        if 'picks' not in picks_data:
            print(f"Could not fetch team for manager ID {manager_id}. It might be an invalid ID.")
            return
            
        team_picks = picks_data['picks']

        # 5. Display the team
        print(f"Your FPL Team for Manager ID: {manager_id}\n" + "="*30)
        starting_lineup = []
        bench = []

        for pick in team_picks:
            player_id = pick['element']
            player_name = player_map.get(player_id, f"Unknown Player (ID: {player_id})")
            
            # Add captain/vice-captain markers
            if pick['is_captain']:
                player_name += " (C)"
            elif pick['is_vice_captain']:
                player_name += " (VC)"
            
            if pick['position'] <= 11:
                starting_lineup.append(player_name)
            else:
                bench.append(player_name)

        print("--- Starting XI ---")
        for player in starting_lineup:
            print(f"- {player}")

        print("\n--- Bench ---")
        for player in bench:
            print(f"- {player}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred with the network request: {e}")
    except KeyError as e:
        print(f"Could not parse the data from the FPL API. Key not found: {e}")


if __name__ == "__main__":
    # 🚨 REPLACE THIS WITH YOUR MANAGER ID 🚨
    my_manager_id = '1897520'  # Example ID for the overall FPL winner of 2019/20
    get_fpl_team(my_manager_id)