# File path: src/ingestion/parse_matches.py
import glob
import json
import os
import sqlite3
from tqdm import tqdm

DB_PATH = os.path.join("data", "processed", "cricket_dss.db")
RAW_JSON_DIR = os.path.join("data", "raw_json")

def create_tables(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dim_players (
        player_id TEXT PRIMARY KEY,
        player_name TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dim_matches (
        match_id TEXT PRIMARY KEY,
        match_type TEXT,
        gender TEXT,
        season TEXT,
        match_date TEXT,
        venue TEXT,
        city TEXT,
        team_1 TEXT,
        team_2 TEXT,
        toss_winner TEXT,
        toss_decision TEXT,
        winner TEXT,
        outcome_method TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fact_deliveries (
        delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT,
        innings_num INTEGER,
        batting_team TEXT,
        bowling_team TEXT,
        over_num INTEGER,
        ball_in_over INTEGER,
        phase TEXT,
        batter_id TEXT,
        batter_name TEXT,
        bowler_id TEXT,
        bowler_name TEXT,
        non_striker_id TEXT,
        runs_batter INTEGER,
        runs_extras INTEGER,
        runs_total INTEGER,
        is_legal_ball INTEGER,
        is_wide INTEGER,
        is_noball INTEGER,
        is_bye INTEGER,
        is_legbye INTEGER,
        is_wicket INTEGER,
        dismissal_type TEXT,
        player_dismissed_id TEXT,
        fielder_id TEXT,
        FOREIGN KEY(match_id) REFERENCES dim_matches(match_id)
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deliv_match ON fact_deliveries(match_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deliv_batter ON fact_deliveries(batter_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deliv_bowler ON fact_deliveries(bowler_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deliv_phase ON fact_deliveries(phase);")

def get_match_phase(match_type: str, over_num: int) -> str:
    """Calculates match phase based on format and zero-indexed over number."""
    mtype = str(match_type).lower()
    if "t20" in mtype or "hundred" in mtype:
        if over_num < 6:
            return "Powerplay"
        elif over_num < 16:
            return "Middle"
        else:
            return "Death"
    else:  # ODI / 50-over match
        if over_num < 10:
            return "Powerplay"
        elif over_num < 40:
            return "Middle"
        else:
            return "Death"

def parse_json_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    match_id = os.path.basename(file_path).replace(".json", "")
    
    # 1. Extract Player Registry from match info
    people_map = info.get("registry", {}).get("people", {})
    players_batch = [(pid, name) for name, pid in people_map.items()]

    # 2. Extract Match Meta
    teams = info.get("teams", ["Unknown", "Unknown"])
    team_1 = teams[0] if len(teams) > 0 else "Unknown"
    team_2 = teams[1] if len(teams) > 1 else "Unknown"
    dates = info.get("dates", [""])
    outcome = info.get("outcome", {})
    
    match_row = (
        match_id,
        info.get("match_type", "Unknown"),
        info.get("gender", "female"),
        str(info.get("season", "")),
        dates[0] if dates else "",
        info.get("venue", "Unknown"),
        info.get("city", "Unknown"),
        team_1,
        team_2,
        info.get("toss", {}).get("winner", "Unknown"),
        info.get("toss", {}).get("decision", "Unknown"),
        outcome.get("winner", outcome.get("result", "No Result")),
        outcome.get("method", "Normal")
    )

    # 3. Extract Deliveries
    deliveries_batch = []
    innings_list = data.get("innings", [])

    for inn_idx, innings_data in enumerate(innings_list, start=1):
        batting_team = innings_data.get("team", "Unknown")
        bowling_team = team_2 if batting_team == team_1 else team_1
        overs = innings_data.get("overs", [])

        for over_data in overs:
            over_num = over_data.get("over", 0)  # 0-indexed in Cricsheet
            phase = get_match_phase(info.get("match_type", "T20"), over_num)
            deliveries = over_data.get("deliveries", [])

            for ball_idx, d in enumerate(deliveries, start=1):
                batter_name = d.get("batter", "")
                bowler_name = d.get("bowler", "")
                non_striker_name = d.get("non_striker", "")

                batter_id = people_map.get(batter_name, batter_name)
                bowler_id = people_map.get(bowler_name, bowler_name)
                non_striker_id = people_map.get(non_striker_name, non_striker_name)

                runs = d.get("runs", {})
                runs_batter = runs.get("batter", 0)
                runs_extras = runs.get("extras", 0)
                runs_total = runs.get("total", 0)

                extras = d.get("extras", {})
                is_wide = 1 if "wides" in extras else 0
                is_noball = 1 if "noballs" in extras else 0
                is_bye = 1 if "byes" in extras else 0
                is_legbye = 1 if "legbyes" in extras else 0
                is_legal = 0 if (is_wide or is_noball) else 1

                wickets = d.get("wickets", [])
                is_wicket = 1 if len(wickets) > 0 else 0
                dismissal_type = ""
                player_dismissed_id = ""
                fielder_id = ""

                if is_wicket:
                    w = wickets[0]
                    dismissal_type = w.get("kind", "")
                    player_dismissed_name = w.get("player_out", "")
                    player_dismissed_id = people_map.get(player_dismissed_name, player_dismissed_name)
                    
                    fielders = w.get("fielders", [])
                    if fielders:
                        f_name = fielders[0].get("name", "")
                        fielder_id = people_map.get(f_name, f_name)

                deliveries_batch.append((
                    match_id,
                    inn_idx,
                    batting_team,
                    bowling_team,
                    over_num,
                    ball_idx,
                    phase,
                    batter_id,
                    batter_name,
                    bowler_id,
                    bowler_name,
                    non_striker_id,
                    runs_batter,
                    runs_extras,
                    runs_total,
                    is_legal,
                    is_wide,
                    is_noball,
                    is_bye,
                    is_legbye,
                    is_wicket,
                    dismissal_type,
                    player_dismissed_id,
                    fielder_id
                ))

    return players_batch, match_row, deliveries_batch

def run_pipeline():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enable Write-Ahead Logging for high-throughput batch inserts
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    
    create_tables(cursor)
    conn.commit()

    all_files = glob.glob(os.path.join(RAW_JSON_DIR, "*.json"))
    print(f"Total match files found: {len(all_files)}")

    insert_player_sql = "INSERT OR IGNORE INTO dim_players (player_id, player_name) VALUES (?, ?);"
    insert_match_sql = """
    INSERT OR REPLACE INTO dim_matches (
        match_id, match_type, gender, season, match_date, venue, city,
        team_1, team_2, toss_winner, toss_decision, winner, outcome_method
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    insert_delivery_sql = """
    INSERT INTO fact_deliveries (
        match_id, innings_num, batting_team, bowling_team, over_num, ball_in_over,
        phase, batter_id, batter_name, bowler_id, bowler_name, non_striker_id,
        runs_batter, runs_extras, runs_total, is_legal_ball, is_wide, is_noball,
        is_bye, is_legbye, is_wicket, dismissal_type, player_dismissed_id, fielder_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    player_set = set()
    match_list = []
    delivery_list = []

    for file_path in tqdm(all_files, desc="Parsing Match Files"):
        try:
            players, match, deliveries = parse_json_file(file_path)
            for p in players:
                player_set.add(p)
            match_list.append(match)
            delivery_list.extend(deliveries)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    print("\nWriting to SQLite database...")
    cursor.executemany(insert_player_sql, list(player_set))
    cursor.executemany(insert_match_sql, match_list)
    cursor.executemany(insert_delivery_sql, delivery_list)
    
    conn.commit()
    conn.close()
    print(f"Ingestion complete! Database saved at: {DB_PATH}")

if __name__ == "__main__":
    run_pipeline()