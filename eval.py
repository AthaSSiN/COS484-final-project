import os
import json
import pandas as pd
from collections import defaultdict

# Define the lists (add more elements as needed)
player_1_models = ["4_1_nano", "4_1_nano", "sapar_4_1_nano"]
player_2_models = ["4_1_mini", "4o_mini", "cg_4_1_mini"]
player_1_sides = ["evil", "good"]
max_indices = [4, 4, 9]

base_log_dir = "/home/atharv/courses/LLM-Game-Agent/playing_log/avalon/battle"
# Structure: {(p1_model, p2_model): {p1_good_wins: 0, p1_evil_wins: 0, p2_good_wins: 0, p2_evil_wins: 0, total_games: 0}}
matchup_results = defaultdict(lambda: defaultdict(int))

# Iterate through all combinations
for i, p1_model in enumerate(player_1_models):
    p2_model = player_2_models[i]
    matchup_key = (p1_model, p2_model)

    for p1_side in player_1_sides:
        p2_side = "good" if p1_side == "evil" else "evil"
        for index in range(max_indices[i] + 1): # Iterate up to the max index provided
            # Construct folder name
            folder_name = f"{p1_model}_{p1_side}_{p2_model}_{p2_side}-{p1_side}-game_{index}" # Check if this format is correct
            folder_path = os.path.join(base_log_dir, folder_name)

            # Check if folder exists
            if os.path.isdir(folder_path):
                json_path = os.path.join(folder_path, "process.json")

                # Check if process.json exists
                if os.path.isfile(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            # Check if data is a non-empty dictionary
                            if data and isinstance(data, dict) and data:
                                # Get the key for the last round (assuming keys are ordered chronologically)
                                last_round_key = list(data.keys())[-1]
                                last_round_events = data[last_round_key]

                                # Check if the last round's event list is valid
                                if last_round_events and isinstance(last_round_events, list) and len(last_round_events) > 0:
                                    # Get the last event dictionary in the last round's list
                                    last_event = last_round_events[-1]

                                    # Check if the last event has a "Host" key
                                    if isinstance(last_event, dict) and "Host" in last_event:
                                        host_message = last_event["Host"]
                                        winner_side = None
                                        if isinstance(host_message, str):
                                            if "Good" in host_message:
                                                winner_side = "good"
                                            elif "Evil" in host_message:
                                                winner_side = "evil"
                                        if winner_side:
                                            # Increment total games for this matchup
                                            matchup_results[matchup_key]['total_games'] += 1

                                            # Attribute win
                                            if p1_side == winner_side:
                                                if winner_side == "good":
                                                    matchup_results[matchup_key]['p1_good_wins'] += 1
                                                else: # evil
                                                    matchup_results[matchup_key]['p1_evil_wins'] += 1
                                            elif p2_side == winner_side:
                                                 if winner_side == "good":
                                                    matchup_results[matchup_key]['p2_good_wins'] += 1
                                                 else: # evil
                                                    matchup_results[matchup_key]['p2_evil_wins'] += 1
                                    else:
                                        print(f"Warning: Last event in {json_path} is not a dict or missing 'Host' key: {last_event}")
                                else:
                                    print(f"Warning: Invalid or empty event list for last round '{last_round_key}' in {json_path}")
                            else:
                                    print(f"Warning: Invalid, empty, or non-dict data in {json_path}")

                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in {json_path}")
                    except Exception as e:
                        print(f"An error occurred processing {json_path}: {e}")
                # else:
                    # print(f"process.json not found in {folder_path}") # Optional
            # else:
            #     print(f"Folder not found: {folder_path}") # Optional: uncomment to see missing folders

# Prepare data for DataFrame
data_for_df = []
for (p1_model, p2_model), stats in matchup_results.items():
    print(f"Matchup: {p1_model} vs {p2_model}")
    p1_total_wins = stats.get('p1_good_wins', 0) + stats.get('p1_evil_wins', 0)
    p2_total_wins = stats.get('p2_good_wins', 0) + stats.get('p2_evil_wins', 0)

    overall_winner = "Draw"
    if p1_total_wins > p2_total_wins:
        overall_winner = p1_model
    elif p2_total_wins > p1_total_wins:
        overall_winner = p2_model

    data_for_df.append({
        'Player 1': p1_model,
        'Player 2': p2_model,
        'P1 Wins as Good': stats.get('p1_good_wins', 0),
        'P1 Wins as Evil': stats.get('p1_evil_wins', 0),
        # 'P2 Wins as Good': stats.get('p2_good_wins', 0), # Optional: uncomment if needed
        # 'P2 Wins as Evil': stats.get('p2_evil_wins', 0), # Optional: uncomment if needed
        'Total Games': stats.get('total_games', 0),
        'Overall Winner': overall_winner
    })

# Create Pandas DataFrame
df = pd.DataFrame(data_for_df)

# Display the DataFrame
print("Matchup Win Counts:")
print(df.to_string(index=False))