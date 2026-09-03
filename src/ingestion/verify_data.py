# File path: src/ingestion/verify_data.py
import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join("data", "processed", "cricket_dss.db")

def verify_database():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. Run parse_matches.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # 1. Row counts
    print("--- 1. Table Counts ---")
    tables = ["dim_players", "dim_matches", "fact_deliveries"]
    for t in tables:
        count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {t};", conn).iloc[0]["count"]
        print(f"{t}: {count:,} records")

    # 2. Phase Distribution
    print("\n--- 2. Deliveries per Phase ---")
    df_phase = pd.read_sql_query("""
        SELECT phase, COUNT(*) as delivery_count, SUM(runs_total) as total_runs, SUM(is_wicket) as total_wickets
        FROM fact_deliveries
        GROUP BY phase;
    """, conn)
    print(df_phase.to_string(index=False))

    # 3. Top 5 Run Scorers Sample Check
    print("\n--- 3. Top 5 All-Rounders / Batters by Total Runs ---")
    df_top_batters = pd.read_sql_query("""
        SELECT batter_name, COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) as balls_faced,
               SUM(runs_batter) as total_runs,
               ROUND(CAST(SUM(runs_batter) AS FLOAT) / COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END) * 100, 2) as strike_rate
        FROM fact_deliveries
        GROUP BY batter_id
        HAVING balls_faced > 500
        ORDER BY total_runs DESC
        LIMIT 5;
    """, conn)
    print(df_top_batters.to_string(index=False))

    conn.close()

if __name__ == "__main__":
    verify_database()