# File path: src/features/venue_opponent_profiles.py
import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = os.path.join("data", "processed", "cricket_dss.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def classify_format(match_type: str) -> str:
    """Standardizes match types into two primary formats: T20 or ODI."""
    m = str(match_type).lower()
    if "t20" in m or "hundred" in m:
        return "T20"
    elif "odi" in m or "50" in m:
        return "ODI"
    return "T20"

def compute_venue_phase_benchmarks(conn):
    """
    Calculates expected par scoring rate and wicket probability for each
    venue and phase (Powerplay, Middle, Death) with Bayesian smoothing.
    """
    print("Computing venue phase benchmarks...")
    
    query = """
    SELECT 
        m.match_id,
        m.match_type,
        m.venue,
        f.phase,
        f.runs_total,
        f.is_legal_ball,
        f.is_wicket
    FROM fact_deliveries f
    JOIN dim_matches m ON f.match_id = m.match_id
    WHERE m.venue IS NOT NULL AND m.venue != 'Unknown';
    """
    df = pd.read_sql_query(query, conn)
    df["format"] = df["match_type"].apply(classify_format)

    # 1. Format & Phase Global Baselines (Fallback prior)
    global_phase = df.groupby(["format", "phase"]).agg(
        global_legal_balls=("is_legal_ball", "sum"),
        global_runs=("runs_total", "sum"),
        global_wickets=("is_wicket", "sum")
    ).reset_index()

    global_phase["global_sr"] = (global_phase["global_runs"] / global_phase["global_legal_balls"]) * 100
    global_phase["global_wicket_prob"] = global_phase["global_wickets"] / global_phase["global_legal_balls"]

    # 2. Raw Venue & Phase Aggregation
    venue_phase = df.groupby(["venue", "format", "phase"]).agg(
        sample_balls=("is_legal_ball", "sum"),
        venue_runs=("runs_total", "sum"),
        venue_wickets=("is_wicket", "sum")
    ).reset_index()

    # Merge global priors
    merged = pd.merge(venue_phase, global_phase, on=["format", "phase"], how="left")

    # 3. Bayesian Shrinkage Smoothing
    # Confidence weight C represents pseudo-deliveries (e.g., 120 balls / 20 overs)
    C = 120.0
    merged["smoothed_sr"] = (
        (merged["venue_runs"] + (merged["global_sr"] / 100.0) * C) / 
        (merged["sample_balls"] + C)
    ) * 100.0

    merged["smoothed_wicket_prob"] = (
        (merged["venue_wickets"] + merged["global_wicket_prob"] * C) / 
        (merged["sample_balls"] + C)
    )

    merged["par_economy"] = (merged["smoothed_sr"] / 100.0) * 6.0

    output_df = merged[[
        "venue", "format", "phase", "sample_balls",
        "smoothed_sr", "par_economy", "smoothed_wicket_prob"
    ]].rename(columns={
        "smoothed_sr": "par_strike_rate",
        "smoothed_wicket_prob": "par_wicket_prob"
    })

    # Save to SQLite
    output_df.to_sql("venue_phase_benchmarks", conn, if_exists="replace", index=False)
    
    # Save Global Baselines for unknown venues
    global_phase_out = global_phase[["format", "phase", "global_sr", "global_wicket_prob"]].rename(
        columns={"global_sr": "par_strike_rate", "global_wicket_prob": "par_wicket_prob"}
    )
    global_phase_out["par_economy"] = (global_phase_out["par_strike_rate"] / 100.0) * 6.0
    global_phase_out["venue"] = "GLOBAL_DEFAULT"
    global_phase_out["sample_balls"] = global_phase["global_legal_balls"]
    global_phase_out.to_sql("venue_phase_benchmarks", conn, if_exists="append", index=False)

    print(f" Saved {len(output_df)} venue-phase benchmark records.")
    return output_df

def compute_team_opponent_ratings(conn):
    """
    Computes batting and bowling strength indices for each team relative
    to the global average across the dataset.
    """
    print("Computing team opponent strength indices (OSI)...")
    
    query = """
    SELECT 
        m.match_type,
        f.batting_team,
        f.bowling_team,
        f.runs_total,
        f.is_legal_ball,
        f.is_wicket
    FROM fact_deliveries f
    JOIN dim_matches m ON f.match_id = m.match_id;
    """
    df = pd.read_sql_query(query, conn)
    df["format"] = df["match_type"].apply(classify_format)

    # Batting metrics per team
    bat_df = df.groupby(["batting_team", "format"]).agg(
        bat_balls=("is_legal_ball", "sum"),
        runs_scored=("runs_total", "sum"),
        wickets_lost=("is_wicket", "sum")
    ).reset_index().rename(columns={"batting_team": "team"})

    # Bowling metrics per team
    bowl_df = df.groupby(["bowling_team", "format"]).agg(
        bowl_balls=("is_legal_ball", "sum"),
        runs_conceded=("runs_total", "sum"),
        wickets_taken=("is_wicket", "sum")
    ).reset_index().rename(columns={"bowling_team": "team"})

    team_df = pd.merge(bat_df, bowl_df, on=["team", "format"], how="outer").fillna(0)

    # Filter out teams with negligible sample sizes (< 120 balls in format)
    team_df = team_df[(team_df["bat_balls"] >= 120) & (team_df["bowl_balls"] >= 120)].copy()

    # Calculate global averages per format
    format_stats = team_df.groupby("format").agg(
        total_runs_scored=("runs_scored", "sum"),
        total_bat_balls=("bat_balls", "sum"),
        total_runs_conceded=("runs_conceded", "sum"),
        total_bowl_balls=("bowl_balls", "sum"),
        total_wickets_taken=("wickets_taken", "sum"),
        total_wickets_lost=("wickets_lost", "sum")
    ).reset_index()

    format_stats["global_bat_rr"] = (format_stats["total_runs_scored"] / format_stats["total_bat_balls"]) * 6.0
    format_stats["global_bowl_sr"] = format_stats["total_bowl_balls"] / np.maximum(format_stats["total_wickets_taken"], 1)

    team_df = pd.merge(team_df, format_stats[["format", "global_bat_rr", "global_bowl_sr"]], on="format", how="left")

    # Team Batting Strength: (Team Run Rate / Global Run Rate)
    team_df["team_bat_rr"] = (team_df["runs_scored"] / team_df["bat_balls"]) * 6.0
    team_df["batting_strength_index"] = team_df["team_bat_rr"] / team_df["global_bat_rr"]

    # Team Bowling Strength: (Global Bowling SR / Team Bowling SR) * (Global Bowl Econ / Team Bowl Econ)
    team_df["team_bowl_econ"] = (team_df["runs_conceded"] / team_df["bowl_balls"]) * 6.0
    team_df["team_bowl_sr"] = team_df["bowl_balls"] / np.maximum(team_df["wickets_taken"], 1)
    
    # Bowling index: lower economy & lower strike rate = higher index
    team_df["bowling_strength_index"] = (
        (team_df["global_bat_rr"] / np.maximum(team_df["team_bowl_econ"], 1e-3)) * 0.5 +
        (team_df["global_bowl_sr"] / np.maximum(team_df["team_bowl_sr"], 1e-3)) * 0.5
    )

    # Normalize indices around 1.0 with a soft bounded range [0.70, 1.30]
    team_df["batting_strength_index"] = team_df["batting_strength_index"].clip(0.70, 1.30)
    team_df["bowling_strength_index"] = team_df["bowling_strength_index"].clip(0.70, 1.30)

    output_team = team_df[[
        "team", "format", "bat_balls", "bowl_balls",
        "batting_strength_index", "bowling_strength_index"
    ]]

    output_team.to_sql("team_opponent_ratings", conn, if_exists="replace", index=False)
    print(f" Saved {len(output_team)} team rating profiles.")
    return output_team

def run_profiles_pipeline():
    conn = get_connection()
    try:
        compute_venue_phase_benchmarks(conn)
        compute_team_opponent_ratings(conn)
        
        # Quick sanity check print
        print("\n--- Top 5 Bowling Attacks in T20 (Opposition Multipliers) ---")
        top_bowl = pd.read_sql_query("""
            SELECT team, format, bowl_balls, ROUND(bowling_strength_index, 3) as bowl_index
            FROM team_opponent_ratings
            WHERE format = 'T20'
            ORDER BY bowling_strength_index DESC
            LIMIT 5;
        """, conn)
        print(top_bowl.to_string(index=False))

        print("\n--- Sample Venue Par Ratings (Powerplay vs Death) ---")
        venue_sample = pd.read_sql_query("""
            SELECT venue, phase, ROUND(par_strike_rate, 2) as par_sr, ROUND(par_economy, 2) as par_econ
            FROM venue_phase_benchmarks
            WHERE format = 'T20' AND venue IN ('Melbourne Cricket Ground', 'Wankhede Stadium', 'Brabourne Stadium')
            ORDER BY venue, phase;
        """, conn)
        print(venue_sample.to_string(index=False))

    finally:
        conn.close()

if __name__ == "__main__":
    run_profiles_pipeline()