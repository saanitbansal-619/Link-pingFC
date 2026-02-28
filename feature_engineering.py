"""
Feature engineering script for Master_Rivals_Combined.csv.
Creates derived metrics for attacking, midfield, defensive, and transition phases.
Original data is never modified; results are saved to a new file.
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Paths (script runs from its own directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "Master_Rivals_Combined.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "Master_Rivals_FeatureEngineered.csv")


def safe_divide(numerator, denominator):
    """
    Safe division: returns 0 when denominator is 0/NaN or when result is NaN/inf.
    """
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return 0
    result = numerator / denominator
    if pd.isna(result) or np.isinf(result):
        return 0
    return result


def _has_cols(df, cols):
    """Return True only if all columns exist in df."""
    return all(c in df.columns for c in cols)


def _replace_nan_inf(series):
    """Replace NaN and inf in a series with 0."""
    return series.replace([np.nan, np.inf, -np.inf], 0)


def main():
    # Load master data and work on a copy (original remains untouched)
    df = pd.read_csv(INPUT_PATH)
    df = df.copy()

    engineered = []

    # -----------------------------------------------------------------------
    # ATTACKING
    # -----------------------------------------------------------------------
    if _has_cols(df, ["Goals_Scored", "xG"]):
        df["Finishing_Efficiency"] = df.apply(
            lambda r: safe_divide(r["Goals_Scored"], r["xG"]), axis=1
        )
        df["Finishing_Efficiency"] = _replace_nan_inf(df["Finishing_Efficiency"])
        engineered.append("Finishing_Efficiency")

    if _has_cols(df, ["xG", "Total Shots"]):
        df["Shot_Quality"] = df.apply(
            lambda r: safe_divide(r["xG"], r["Total Shots"]), axis=1
        )
        df["Shot_Quality"] = _replace_nan_inf(df["Shot_Quality"])
        engineered.append("Shot_Quality")

    if _has_cols(df, ["Goals_Scored", "Touches in penalty area"]):
        df["Box_Conversion"] = df.apply(
            lambda r: safe_divide(r["Goals_Scored"], r["Touches in penalty area"]),
            axis=1,
        )
        df["Box_Conversion"] = _replace_nan_inf(df["Box_Conversion"])
        engineered.append("Box_Conversion")

    if _has_cols(df, ["Offensive Duels Won", "Total Offensive Duels"]):
        df["Offensive_Duel_Win_Rate"] = df.apply(
            lambda r: safe_divide(r["Offensive Duels Won"], r["Total Offensive Duels"]),
            axis=1,
        )
        df["Offensive_Duel_Win_Rate"] = _replace_nan_inf(df["Offensive_Duel_Win_Rate"])
        engineered.append("Offensive_Duel_Win_Rate")

    if _has_cols(df, ["Accurate Crosses", "Total Crosses"]):
        df["Cross_Accuracy"] = df.apply(
            lambda r: safe_divide(r["Accurate Crosses"], r["Total Crosses"]), axis=1
        )
        df["Cross_Accuracy"] = _replace_nan_inf(df["Cross_Accuracy"])
        engineered.append("Cross_Accuracy")

    # -----------------------------------------------------------------------
    # MIDFIELD
    # -----------------------------------------------------------------------
    if _has_cols(df, ["Total Forward Passes", "Total Passes"]):
        df["Verticality_Index"] = df.apply(
            lambda r: safe_divide(r["Total Forward Passes"], r["Total Passes"]), axis=1
        )
        df["Verticality_Index"] = _replace_nan_inf(df["Verticality_Index"])
        engineered.append("Verticality_Index")

    if _has_cols(df, ["Total Progressive Passes", "Total Passes"]):
        df["Progressive_Pass_Rate"] = df.apply(
            lambda r: safe_divide(r["Total Progressive Passes"], r["Total Passes"]),
            axis=1,
        )
        df["Progressive_Pass_Rate"] = _replace_nan_inf(df["Progressive_Pass_Rate"])
        engineered.append("Progressive_Pass_Rate")

    if _has_cols(
        df,
        ["Accurate Passes To The Final Third", "Total Passes To The Final Third"],
    ):
        df["Final_Third_Pass_Accuracy"] = df.apply(
            lambda r: safe_divide(
                r["Accurate Passes To The Final Third"],
                r["Total Passes To The Final Third"],
            ),
            axis=1,
        )
        df["Final_Third_Pass_Accuracy"] = _replace_nan_inf(
            df["Final_Third_Pass_Accuracy"]
        )
        engineered.append("Final_Third_Pass_Accuracy")

    if _has_cols(df, ["Accurate Passes", "Total Passes"]):
        df["Possession_Control"] = df.apply(
            lambda r: safe_divide(r["Accurate Passes"], r["Total Passes"]), axis=1
        )
        df["Possession_Control"] = _replace_nan_inf(df["Possession_Control"])
        engineered.append("Possession_Control")

    # -----------------------------------------------------------------------
    # DEFENSIVE
    # -----------------------------------------------------------------------
    if _has_cols(df, ["Defensive Duels Won", "Total Defensive Duels"]):
        df["Defensive_Duel_Win_Rate"] = df.apply(
            lambda r: safe_divide(
                r["Defensive Duels Won"], r["Total Defensive Duels"]
            ),
            axis=1,
        )
        df["Defensive_Duel_Win_Rate"] = _replace_nan_inf(
            df["Defensive_Duel_Win_Rate"]
        )
        engineered.append("Defensive_Duel_Win_Rate")

    if _has_cols(df, ["Aerial Duels Won", "Total Aerial Duels"]):
        df["Aerial_Duel_Win_Rate"] = df.apply(
            lambda r: safe_divide(r["Aerial Duels Won"], r["Total Aerial Duels"]),
            axis=1,
        )
        df["Aerial_Duel_Win_Rate"] = _replace_nan_inf(df["Aerial_Duel_Win_Rate"])
        engineered.append("Aerial_Duel_Win_Rate")

    if _has_cols(df, ["Total Shots Against"]):
        df["Shot_Suppression"] = df.apply(
            lambda r: safe_divide(1, r["Total Shots Against"]), axis=1
        )
        df["Shot_Suppression"] = _replace_nan_inf(df["Shot_Suppression"])
        engineered.append("Shot_Suppression")

    if _has_cols(df, ["Shots Against On Target", "Total Shots Against"]):
        df["Shots_On_Target_Against_Rate"] = df.apply(
            lambda r: safe_divide(
                r["Shots Against On Target"], r["Total Shots Against"]
            ),
            axis=1,
        )
        df["Shots_On_Target_Against_Rate"] = _replace_nan_inf(
            df["Shots_On_Target_Against_Rate"]
        )
        engineered.append("Shots_On_Target_Against_Rate")

    # -----------------------------------------------------------------------
    # ATTACKING TRANSITION
    # -----------------------------------------------------------------------
    if _has_cols(df, ["High Recoveries", "Total Recoveries"]):
        df["High_Recovery_Rate"] = df.apply(
            lambda r: safe_divide(r["High Recoveries"], r["Total Recoveries"]), axis=1
        )
        df["High_Recovery_Rate"] = _replace_nan_inf(df["High_Recovery_Rate"])
        engineered.append("High_Recovery_Rate")

    if _has_cols(df, ["PPDA"]):
        df["Pressing_Intensity"] = df.apply(
            lambda r: safe_divide(1, r["PPDA"]), axis=1
        )
        df["Pressing_Intensity"] = _replace_nan_inf(df["Pressing_Intensity"])
        engineered.append("Pressing_Intensity")

    # -----------------------------------------------------------------------
    # DEFENSIVE TRANSITION
    # -----------------------------------------------------------------------
    if _has_cols(df, ["High Losses", "Total Losses"]):
        df["High_Loss_Rate"] = df.apply(
            lambda r: safe_divide(r["High Losses"], r["Total Losses"]), axis=1
        )
        df["High_Loss_Rate"] = _replace_nan_inf(df["High_Loss_Rate"])
        engineered.append("High_Loss_Rate")

    if _has_cols(df, ["High Losses", "Total Shots Against"]):
        df["Def_Transition_Exposure"] = df.apply(
            lambda r: safe_divide(r["High Losses"], r["Total Shots Against"]), axis=1
        )
        df["Def_Transition_Exposure"] = _replace_nan_inf(
            df["Def_Transition_Exposure"]
        )
        engineered.append("Def_Transition_Exposure")

    # Save result (original CSV never written)
    df.to_csv(OUTPUT_PATH, index=False)

    # Summary
    print(f"Total number of engineered features created: {len(engineered)}")
    print(f"Final dataframe shape: {df.shape}")
    print("Feature engineering complete.")


if __name__ == "__main__":
    main()
