# File path: src/features/compute_metrics.py
import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = os.path.join("data", "processed", "cricket_dss.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def classify_format(match_type: str) -> str:
    m = str(match_type).lower()
    if "t20" in m or "hundred" in m:
        return "T20"
    elif "odi" in m or "50" in m:
        return "ODI"
    return "T20"

def load_base_data(conn):
    """Loads deliveries joined with match meta, venue benchmarks, and opponent ratings."""
    print("1. Loading deliveries and metadata from database...")
    
    query = """
    SELECT 
        f.delivery_id,
        f.match_id,
        m.match_type,
        m.match_date,
        m.venue,
        f.innings_num,
        f.over_num,
        f.ball_in_over,
        f.phase,
        f.batting_team,
        f.bowling_team,
        f.batter_id,
        f.batter_name,
        f.bowler_id,
        f.bowler_name,
        f.runs_batter,
        f.runs_extras,
        f.runs_total,
        f.is_legal_ball,
        f.is_wide,
        f.is_noball,
        f.is_bye,
        f.is_legbye,
        f.is_wicket,
        f.dismissal_type,
        f.player_dismissed_id,
        f.fielder_id
    FROM fact_deliveries f
    JOIN dim_matches m ON f.match_id = m.match_id;
    """
    df = pd.read_sql_query(query, conn)
    df["format"] = df["match_type"].apply(classify_format)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Load venue benchmarks
    df_venue = pd.read_sql_query("SELECT venue, format, phase, par_strike_rate, par_economy FROM venue_phase_benchmarks;", conn)
    
    # Load opponent ratings
    df_opp = pd.read_sql_query("SELECT team, format, batting_strength_index, bowling_strength_index FROM team_opponent_ratings;", conn)

    # Merge venue benchmarks (defaulting to GLOBAL_DEFAULT if venue not matched)
    df = pd.merge(df, df_venue, on=["venue", "format", "phase"], how="left")
    
    # Fill missing venue par with global defaults
    df_global_venue = df_venue[df_venue["venue"] == "GLOBAL_DEFAULT"].drop(columns=["venue"], errors="ignore")
    df = df.fillna({"par_strike_rate": np.nan, "par_economy": np.nan})
    df = pd.merge(df, df_global_venue, on=["format", "phase"], how="left", suffixes=("", "_global"))
    df["par_strike_rate"] = df["par_strike_rate"].fillna(df["par_strike_rate_global"]).fillna(100.0)
    df["par_economy"] = df["par_economy"].fillna(df["par_economy_global"]).fillna(6.0)
    df = df.drop(columns=["par_strike_rate_global", "par_economy_global"], errors="ignore")

    # Merge opponent bowling strength (evaluates batter performance against bowling attack)
    df = pd.merge(
        df, 
        df_opp[["team", "format", "bowling_strength_index"]].rename(columns={"team": "bowling_team"}), 
        on=["bowling_team", "format"], 
        how="left"
    )
    df["bowling_strength_index"] = df["bowling_strength_index"].fillna(1.0)

    # Merge opponent batting strength (evaluates bowler performance against batting lineup)
    df = pd.merge(
        df, 
        df_opp[["team", "format", "batting_strength_index"]].rename(columns={"team": "batting_team"}), 
        on=["batting_team", "format"], 
        how="left"
    )
    df["batting_strength_index"] = df["batting_strength_index"].fillna(1.0)

    return df

def assign_batting_positions(df):
    """Derives batting lineup position (1 to 11) dynamically based on innings entry order."""
    print("2. Assigning dynamic batting positions per innings...")
    
    df_sorted = df.sort_values(by=["match_id", "innings_num", "over_num", "ball_in_over"]).copy()
    
    # Order of batters appearing at crease
    batters_entry = df_sorted.groupby(["match_id", "innings_num", "batter_id"], sort=False).first().reset_index()
    batters_entry["batting_pos"] = batters_entry.groupby(["match_id", "innings_num"]).cumcount() + 1
    
    df_merged = pd.merge(
        df_sorted, 
        batters_entry[["match_id", "innings_num", "batter_id", "batting_pos"]], 
        on=["match_id", "innings_num", "batter_id"], 
        how="left"
    )
    
    # Wicket quality multiplier: Top-order (1-4) = 1.30, Middle (5-7) = 1.0, Tail (8+) = 0.60
    pos = df_merged["batting_pos"].fillna(7)
    df_merged["wicket_weight"] = np.where(pos <= 4, 1.30, np.where(pos <= 7, 1.00, 0.60))
    
    return df_merged

def compute_batting_features(df, target_format="T20", min_balls=150):
    """Computes overall and phase-specific context-adjusted batting metrics."""
    print(f"3. Computing Batting features for {target_format}...")
    df_fmt = df[df["format"] == target_format].copy()

    legal_df = df_fmt[df_fmt["is_legal_ball"] == 1].copy()
    legal_df["weighted_runs"] = legal_df["runs_batter"] * legal_df["bowling_strength_index"]
    legal_df["is_dot"] = (legal_df["runs_batter"] == 0).astype(int)
    legal_df["is_boundary"] = legal_df["runs_batter"].isin([4, 6]).astype(int)
    legal_df["is_four"] = (legal_df["runs_batter"] == 4).astype(int)
    legal_df["is_six"] = (legal_df["runs_batter"] == 6).astype(int)
    legal_df["expected_runs"] = legal_df["par_strike_rate"] / 100.0

    grouped = legal_df.groupby(["batter_id", "batter_name"]).agg(
        total_balls_faced=("is_legal_ball", "sum"),
        total_runs_scored=("runs_batter", "sum"),
        weighted_runs_scored=("weighted_runs", "sum"),
        dots_faced=("is_dot", "sum"),
        boundaries_scored=("is_boundary", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        expected_runs=("expected_runs", "sum")
    ).reset_index()

    # Dismissal counts
    dismissals = df_fmt[
        (df_fmt["is_wicket"] == 1) & 
        (~df_fmt["dismissal_type"].isin(["retired hurt", "obstructing the field"]))
    ].groupby("player_dismissed_id").agg(dismissals=("is_wicket", "sum")).reset_index()

    grouped = pd.merge(grouped, dismissals, left_on="batter_id", right_on="player_dismissed_id", how="left").fillna({"dismissals": 0})
    grouped = grouped[grouped["total_balls_faced"] >= min_balls].copy()

    # Core Batting Metrics
    grouped["raw_bat_sr"] = (grouped["total_runs_scored"] / grouped["total_balls_faced"]) * 100.0
    grouped["raw_bat_avg"] = grouped["total_runs_scored"] / np.maximum(grouped["dismissals"], 1)
    
    # True Strike Rate (TSR) = Actual SR - Expected Venue Par SR
    grouped["expected_sr"] = (grouped["expected_runs"] / grouped["total_balls_faced"]) * 100.0
    grouped["bat_tsr"] = grouped["raw_bat_sr"] - grouped["expected_sr"]
    
    # Context-Adjusted Batting Average (runs weighted by opposition bowling strength)
    grouped["context_bat_avg"] = grouped["weighted_runs_scored"] / np.maximum(grouped["dismissals"], 1)
    grouped["bat_dot_pct"] = (grouped["dots_faced"] / grouped["total_balls_faced"]) * 100.0
    grouped["bat_boundary_pct"] = (grouped["boundaries_scored"] / grouped["total_balls_faced"]) * 100.0

    # Phase-specific breakdown
    for phase in ["Powerplay", "Middle", "Death"]:
        phase_df = legal_df[legal_df["phase"] == phase]
        phase_grp = phase_df.groupby("batter_id").agg(
            p_balls=("is_legal_ball", "sum"),
            p_runs=("runs_batter", "sum"),
            p_exp_runs=("expected_runs", "sum")
        ).reset_index()
        
        phase_grp[f"bat_tsr_{phase.lower()}"] = (
            (phase_grp["p_runs"] / np.maximum(phase_grp["p_balls"], 1)) * 100.0 - 
            (phase_grp["p_exp_runs"] / np.maximum(phase_grp["p_balls"], 1)) * 100.0
        )
        grouped = pd.merge(
            grouped, 
            phase_grp[["batter_id", f"bat_tsr_{phase.lower()}"]], 
            on="batter_id", 
            how="left"
        ).fillna({f"bat_tsr_{phase.lower()}": 0.0})

    return grouped.drop(columns=["player_dismissed_id", "expected_runs", "expected_sr"], errors="ignore")

def compute_bowling_features(df, target_format="T20", min_balls=120):
    """Computes overall and phase-specific context-adjusted bowling metrics."""
    print(f"4. Computing Bowling features for {target_format}...")
    df_fmt = df[df["format"] == target_format].copy()

    runs_conceded_mask = (df_fmt["is_bye"] == 0) & (df_fmt["is_legbye"] == 0)
    df_fmt["bowler_runs_conceded"] = np.where(runs_conceded_mask, df_fmt["runs_total"], 0)

    valid_wkts = ["bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"]
    df_fmt["is_bowler_wicket"] = np.where((df_fmt["is_wicket"] == 1) & (df_fmt["dismissal_type"].isin(valid_wkts)), 1, 0)
    df_fmt["weighted_wicket_value"] = df_fmt["is_bowler_wicket"] * df_fmt["wicket_weight"] * df_fmt["batting_strength_index"]
    df_fmt["is_bowler_dot"] = np.where((df_fmt["bowler_runs_conceded"] == 0) & (df_fmt["is_legal_ball"] == 1), 1, 0)
    df_fmt["is_bowler_boundary"] = np.where(df_fmt["bowler_runs_conceded"].isin([4, 6]), 1, 0)
    df_fmt["expected_runs_conceded"] = (df_fmt["par_economy"] / 6.0) * df_fmt["is_legal_ball"]

    grouped = df_fmt.groupby(["bowler_id", "bowler_name"]).agg(
        total_balls_bowled=("is_legal_ball", "sum"),
        total_runs_conceded=("bowler_runs_conceded", "sum"),
        total_wickets=("is_bowler_wicket", "sum"),
        wicket_quality_score=("weighted_wicket_value", "sum"),
        dots_bowled=("is_bowler_dot", "sum"),
        boundaries_conceded=("is_bowler_boundary", "sum"),
        expected_runs_conceded=("expected_runs_conceded", "sum")
    ).reset_index()

    grouped = grouped[grouped["total_balls_bowled"] >= min_balls].copy()

    # Core Bowling Metrics
    grouped["overs_bowled"] = grouped["total_balls_bowled"] / 6.0
    grouped["raw_bowl_econ"] = (grouped["total_runs_conceded"] / grouped["total_balls_bowled"]) * 6.0
    grouped["raw_bowl_sr"] = grouped["total_balls_bowled"] / np.maximum(grouped["total_wickets"], 1)
    grouped["raw_bowl_avg"] = grouped["total_runs_conceded"] / np.maximum(grouped["total_wickets"], 1)
    
    # True Economy Rate (TER) = Expected Venue Par Economy - Actual Bowler Economy
    grouped["expected_econ"] = (grouped["expected_runs_conceded"] / grouped["total_balls_bowled"]) * 6.0
    grouped["bowl_ter"] = grouped["expected_econ"] - grouped["raw_bowl_econ"]
    
    # Wicket Quality Index per over
    grouped["wqi_per_over"] = grouped["wicket_quality_score"] / np.maximum(grouped["overs_bowled"], 1.0)
    grouped["bowl_dot_pct"] = (grouped["dots_bowled"] / grouped["total_balls_bowled"]) * 100.0

    # Phase-specific breakdown
    for phase in ["Powerplay", "Middle", "Death"]:
        phase_df = df_fmt[df_fmt["phase"] == phase]
        phase_grp = phase_df.groupby("bowler_id").agg(
            p_balls=("is_legal_ball", "sum"),
            p_runs=("bowler_runs_conceded", "sum"),
            p_wkts=("is_bowler_wicket", "sum"),
            p_exp_runs=("expected_runs_conceded", "sum")
        ).reset_index()
        
        phase_grp[f"bowl_ter_{phase.lower()}"] = (
            (phase_grp["p_exp_runs"] / np.maximum(phase_grp["p_balls"], 1)) * 6.0 -
            (phase_grp["p_runs"] / np.maximum(phase_grp["p_balls"], 1)) * 6.0
        )
        phase_grp[f"bowl_wkts_{phase.lower()}"] = phase_grp["p_wkts"]

        grouped = pd.merge(
            grouped, 
            phase_grp[["bowler_id", f"bowl_ter_{phase.lower()}", f"bowl_wkts_{phase.lower()}"]], 
            on="bowler_id", 
            how="left"
        ).fillna({f"bowl_ter_{phase.lower()}": 0.0, f"bowl_wkts_{phase.lower()}": 0})

    return grouped.drop(columns=["expected_runs_conceded", "expected_econ"], errors="ignore")

def compute_fielding_features(df, target_format="T20"):
    """Aggregates fielding dismissals (catches, stumpings, run outs)."""
    print(f"5. Computing Fielding features for {target_format}...")
    df_fmt = df[df["format"] == target_format].copy()
    
    # Filter only deliveries where a valid fielder_id is recorded
    fielding_wkts = df_fmt[
        (df_fmt["is_wicket"] == 1) & 
        (df_fmt["fielder_id"].notna()) & 
        (df_fmt["fielder_id"] != "")
    ].copy()

    catches = fielding_wkts[fielding_wkts["dismissal_type"].isin(["caught", "caught and bowled"])].groupby("fielder_id").size().rename("catches")
    stumpings = fielding_wkts[fielding_wkts["dismissal_type"] == "stumped"].groupby("fielder_id").size().rename("stumpings")
    run_outs = fielding_wkts[fielding_wkts["dismissal_type"] == "run out"].groupby("fielder_id").size().rename("run_outs")

    df_field = pd.concat([catches, stumpings, run_outs], axis=1).fillna(0).reset_index()
    
    # Rename the first column to player_id regardless of whether index was named fielder_id or index
    first_col = df_field.columns[0]
    df_field = df_field.rename(columns={first_col: "player_id"})
    
    df_field["fielding_impact_score"] = df_field["catches"] * 1.0 + df_field["stumpings"] * 1.5 + df_field["run_outs"] * 1.75
    return df_field

def compute_match_impact_consistency(df, target_format="T20"):
    """
    Calculates match-by-match unified impact score to evaluate longitudinal consistency
    (1 / CV) and Exponentially Weighted Moving Average (EWMA) form.
    """
    print(f"6. Computing match consistency & EWMA form for {target_format}...")
    df_fmt = df[df["format"] == target_format].copy()

    # Per-match Batting Points
    bat_legal = df_fmt[df_fmt["is_legal_ball"] == 1].copy()
    bat_legal["weighted_runs"] = bat_legal["runs_batter"] * bat_legal["bowling_strength_index"]
    bat_legal["exp_runs"] = bat_legal["par_strike_rate"] / 100.0

    bat_match = bat_legal.groupby(["match_id", "match_date", "batter_id"]).agg(
        weighted_runs=("weighted_runs", "sum"),
        exp_runs=("exp_runs", "sum")
    ).reset_index()
    bat_match["bat_impact"] = bat_match["weighted_runs"] - bat_match["exp_runs"]

    # Per-match Bowling Points
    valid_wkts = ["bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"]
    df_fmt["is_bowler_wicket"] = np.where((df_fmt["is_wicket"] == 1) & (df_fmt["dismissal_type"].isin(valid_wkts)), 1, 0)
    df_fmt["runs_conceded"] = np.where((df_fmt["is_bye"] == 0) & (df_fmt["is_legbye"] == 0), df_fmt["runs_total"], 0)
    df_fmt["exp_runs_conceded"] = (df_fmt["par_economy"] / 6.0) * df_fmt["is_legal_ball"]

    bowl_match = df_fmt.groupby(["match_id", "match_date", "bowler_id"]).agg(
        runs_conceded=("runs_conceded", "sum"),
        exp_runs_conceded=("exp_runs_conceded", "sum"),
        wkts=("is_bowler_wicket", "sum"),
        opp_factor=("batting_strength_index", "mean")
    ).reset_index()
    bowl_match["bowl_impact"] = (bowl_match["exp_runs_conceded"] - bowl_match["runs_conceded"]) + (bowl_match["wkts"] * 20.0 * bowl_match["opp_factor"])

    # Combine into unified match events per player
    bat_match = bat_match.rename(columns={"batter_id": "player_id"})[["match_id", "match_date", "player_id", "bat_impact"]]
    bowl_match = bowl_match.rename(columns={"bowler_id": "player_id"})[["match_id", "match_date", "player_id", "bowl_impact"]]

    match_impact = pd.merge(bat_match, bowl_match, on=["match_id", "match_date", "player_id"], how="outer").fillna(0)
    match_impact["total_match_impact"] = match_impact["bat_impact"] + match_impact["bowl_impact"]
    match_impact = match_impact.sort_values(by=["player_id", "match_date"])

    # Compute Stability (1/CV) and EWMA (alpha=0.25)
    records = []
    for pid, group in match_impact.groupby("player_id"):
        impacts = group["total_match_impact"].values
        matches_played = len(impacts)
        if matches_played >= 5:
            mean_impact = np.mean(impacts)
            std_impact = np.std(impacts)
            cv = (std_impact / (mean_impact + 1e-5)) if mean_impact > 0 else 2.0
            stability_score = float(np.clip(1.0 / (abs(cv) + 0.1), 0.1, 5.0))
            ewma_form = float(pd.Series(impacts).ewm(alpha=0.25).mean().iloc[-1])
        else:
            stability_score = 1.0
            ewma_form = float(np.mean(impacts)) if matches_played > 0 else 0.0

        records.append({
            "player_id": pid,
            "matches_tracked": matches_played,
            "stability_score": stability_score,
            "ewma_form_score": ewma_form
        })

    return pd.DataFrame(records)

def run_metrics_pipeline():
    conn = get_connection()
    try:
        raw_df = load_base_data(conn)
        df = assign_batting_positions(raw_df)

        target_fmt = "T20"
        df_bat = compute_batting_features(df, target_format=target_fmt, min_balls=150)
        df_bowl = compute_bowling_features(df, target_format=target_fmt, min_balls=120)
        df_field = compute_fielding_features(df, target_format=target_fmt)
        df_cons = compute_match_impact_consistency(df, target_format=target_fmt)

        print("\n7. Merging into all-rounder consolidated feature matrix...")
        
        # Inner join on players qualifying in both batting and bowling
        ar_df = pd.merge(
            df_bat, 
            df_bowl, 
            left_on="batter_id", 
            right_on="bowler_id", 
            how="inner"
        )
        
        ar_df["player_id"] = ar_df["batter_id"]
        ar_df["player_name"] = ar_df["batter_name"]
        ar_df = ar_df.drop(columns=["batter_id", "bowler_id", "batter_name", "bowler_name"])

        # Merge fielding & consistency
        ar_df = pd.merge(ar_df, df_field, on="player_id", how="left")
        ar_df["catches"] = ar_df["catches"].fillna(0)
        ar_df["stumpings"] = ar_df["stumpings"].fillna(0)
        ar_df["run_outs"] = ar_df["run_outs"].fillna(0)
        ar_df["fielding_impact_score"] = ar_df["fielding_impact_score"].fillna(0.0)

        ar_df = pd.merge(ar_df, df_cons, on="player_id", how="left")
        ar_df["stability_score"] = ar_df["stability_score"].fillna(1.0)
        ar_df["ewma_form_score"] = ar_df["ewma_form_score"].fillna(0.0)
        ar_df["matches_tracked"] = ar_df["matches_tracked"].fillna(0)
        ar_df["format"] = target_fmt

        # Save to SQLite table and Parquet format
        ar_df.to_sql("player_feature_matrix", conn, if_exists="replace", index=False)
        
        parquet_path = os.path.join("data", "processed", "player_feature_matrix.parquet")
        ar_df.to_parquet(parquet_path, index=False)

        print(f"\n Successfully built Feature Matrix for {len(ar_df)} qualifying Women's All-Rounders.")
        print(f" Saved to SQLite table 'player_feature_matrix' and '{parquet_path}'")

        # Snapshot preview
        preview_cols = [
            "player_name", "total_runs_scored", "bat_tsr", "bat_tsr_death",
            "total_wickets", "bowl_ter", "bowl_ter_death", "stability_score", "ewma_form_score"
        ]
        sample = ar_df[preview_cols].sort_values(by="total_runs_scored", ascending=False).head(8)
        print("\n--- Sample All-Rounders Feature Snapshot ---")
        print(sample.to_string(index=False))

    finally:
        conn.close()

if __name__ == "__main__":
    run_metrics_pipeline()