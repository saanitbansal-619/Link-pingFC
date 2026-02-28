import os
import sys
import pandas as pd
import re


def is_hidden_file(filename: str) -> bool:
    """Check if a file should be treated as hidden/system file."""
    basename = os.path.basename(filename)
    return basename.startswith('.') or basename.startswith('~')


def extract_team_name(filename: str) -> str | None:
    """
    Extract team name from filename pattern: "RivalsStats - TeamName.csv"
    Returns None if filename doesn't match the expected pattern.
    """
    basename = os.path.basename(filename)
    
    # Check if filename matches the pattern
    if not basename.startswith("RivalsStats - "):
        return None
    
    # Remove "RivalsStats - " prefix
    team_name = basename.replace("RivalsStats - ", "", 1)
    
    # Remove .csv extension (case-insensitive)
    if team_name.lower().endswith(".csv"):
        team_name = team_name[:-4]
    
    # Strip extra whitespace
    team_name = team_name.strip()
    
    return team_name if team_name else None


def parse_match_column(match_str: str, extracted_team: str) -> tuple[str, int, int] | None:
    """
    Parse Match column format: "Team A - Team B X:Y"
    Returns tuple of (Opponent, Goals_Scored, Goals_Conceded) or None if parsing fails.
    """
    if pd.isna(match_str) or not isinstance(match_str, str):
        return None
    
    match_str = match_str.strip()
    
    # Pattern to match "Team A - Team B X:Y" format
    # Handles extra spaces and variations
    pattern = r'^(.+?)\s*-\s*(.+?)\s+(\d+):(\d+)$'
    match = re.match(pattern, match_str)
    
    if not match:
        return None
    
    team_a = match.group(1).strip()
    team_b = match.group(2).strip()
    goals_a = int(match.group(3))
    goals_b = int(match.group(4))
    
    # Determine which team is the extracted team
    if extracted_team.strip() == team_a:
        return (team_b, goals_a, goals_b)
    elif extracted_team.strip() == team_b:
        return (team_a, goals_b, goals_a)
    else:
        # If extracted team doesn't match either, return None
        return None


def load_csv_with_team(file_path: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Load and process a CSV file:
    - Extract team name from filename
    - Load CSV with UTF-8 encoding
    - Strip whitespace from column names and string values
    - Drop rows 1 and 2 (keep row 0 with headers)
    - Filter to keep only rows where Team column matches extracted team name
    - Parse Match column and create Opponent, Goals_Scored, Goals_Conceded columns
    
    Returns tuple of (DataFrame or None, error_message or None).
    """
    try:
        # Extract team name from filename
        extracted_team = extract_team_name(file_path)
        if extracted_team is None:
            return None, f"Filename doesn't match expected pattern 'RivalsStats - TeamName.csv'"

        # Read CSV with UTF-8 encoding to handle special characters (Örebro, Umeå, Häcken, etc.)
        df = pd.read_csv(file_path, encoding='utf-8')

        # Strip whitespace from all column names
        df.columns = df.columns.str.strip()

        # Drop rows 1 and 2 (positional index 0 and 1) if they exist
        # Row 0 contains headers, so we keep it
        if len(df) >= 2:
            df = df.drop(df.index[[0, 1]])
        elif len(df) >= 1:
            df = df.drop(df.index[0])

        # Reset index after dropping rows
        df = df.reset_index(drop=True)

        # Strip whitespace from all string values in the DataFrame
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                df[col] = df[col].astype(str).str.strip()
                # Replace 'nan' strings back to NaN
                df[col] = df[col].replace('nan', pd.NA)

        # Check if "Team" column exists
        if 'Team' not in df.columns:
            return None, "CSV file does not contain a 'Team' column"

        # Filter: Keep ONLY rows where Team column equals extracted team name
        # Handle potential whitespace differences
        df['Team'] = df['Team'].astype(str).str.strip()
        df = df[df['Team'] == extracted_team.strip()].copy()

        if df.empty:
            return None, f"No rows found where Team column matches '{extracted_team}'"

        # Check if "Match" column exists
        if 'Match' not in df.columns:
            return None, "CSV file does not contain a 'Match' column"

        # Parse Match column and create new columns
        match_results = df['Match'].apply(lambda x: parse_match_column(x, extracted_team))
        
        # Create new columns
        df['Opponent'] = match_results.apply(lambda x: x[0] if x is not None else None)
        df['Goals_Scored'] = match_results.apply(lambda x: x[1] if x is not None else None)
        df['Goals_Conceded'] = match_results.apply(lambda x: x[2] if x is not None else None)

        # Convert Goals_Scored and Goals_Conceded to integers
        # Handle any None/NaN values gracefully
        df['Goals_Scored'] = pd.to_numeric(df['Goals_Scored'], errors='coerce').astype('Int64')
        df['Goals_Conceded'] = pd.to_numeric(df['Goals_Conceded'], errors='coerce').astype('Int64')

        # Drop unnecessary columns: 'Goals:' (or 'Goals'), 'Competition', 'Match'
        columns_to_drop = [col for col in ['Goals:', 'Goals', 'Competition', 'Match'] if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        # Reorder columns so that Opponent, Goals_Scored, Goals_Conceded come
        # immediately after Team, keeping all other stats columns as-is
        if 'Team' in df.columns:
            cols = list(df.columns)
            # Remove the target columns to reinsert them after Team
            for c in ['Opponent', 'Goals_Scored', 'Goals_Conceded']:
                if c in cols:
                    cols.remove(c)
            cols.remove('Team')
            new_order = ['Team', 'Opponent', 'Goals_Scored', 'Goals_Conceded'] + cols
            df = df[[c for c in new_order if c in df.columns]]

        return df, None

    except UnicodeDecodeError as e:
        error_msg = f"Encoding error: {e}"
        return None, error_msg
    except pd.errors.EmptyDataError:
        error_msg = "File is empty"
        return None, error_msg
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        return None, error_msg


def combine_team_csvs(folder_path: str, output_filename: str = "Master_Rivals_Combined.csv") -> None:
    """Combine all CSV files in a folder into a single master CSV."""
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.", file=sys.stderr)
        return

    try:
        # Get all files in the folder (handle special characters in filenames)
        all_files = os.listdir(folder_path)
    except Exception as e:
        print(f"Error reading directory '{folder_path}': {e}", file=sys.stderr)
        return

    # Find all CSV files (excluding hidden/system files)
    csv_files = []
    for f in all_files:
        if f.lower().endswith(".csv") and not is_hidden_file(f):
            csv_files.append(os.path.join(folder_path, f))

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    # Load all CSV files with Team column added
    dataframes = []
    skipped_files = []
    
    for file_path in csv_files:
        df, error_msg = load_csv_with_team(file_path)
        if df is not None and not df.empty:
            dataframes.append(df)
        else:
            filename = os.path.basename(file_path)
            skipped_files.append((filename, error_msg or "Unknown error"))

    if not dataframes:
        print("No valid CSV data loaded. Nothing to combine.")
        if skipped_files:
            print("\nFiles skipped:")
            for filename, error in skipped_files:
                print(f"  - {filename}: {error}")
        return

    # Concatenate all DataFrames vertically
    try:
        master_df = pd.concat(dataframes, axis=0, ignore_index=True)
    except Exception as e:
        print(f"Error concatenating DataFrames: {e}", file=sys.stderr)
        return

    # Reset index
    master_df = master_df.reset_index(drop=True)

    # Save the master dataset to the same folder
    output_path = os.path.join(folder_path, output_filename)
    try:
        master_df.to_csv(output_path, index=False, encoding='utf-8')
    except Exception as e:
        print(f"Error saving combined CSV to '{output_path}': {e}", file=sys.stderr)
        return

    # Print summary information
    print(f"\n{'='*70}")
    print(f"SUCCESS: Combined CSV saved as: {output_path}")
    print(f"{'='*70}")
    print(f"Number of teams processed: {len(dataframes)}")
    
    if skipped_files:
        print(f"\nFiles skipped ({len(skipped_files)}):")
        for filename, error in skipped_files:
            print(f"  - {filename}: {error}")
    
    print(f"\nFinal dataset shape: {master_df.shape[0]} rows × {master_df.shape[1]} columns")
    print(f"\nFirst 5 rows preview:")
    print(f"{'='*70}")
    print(master_df.head().to_string())
    print(f"{'='*70}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python combine_rivals.py <folder_path>")
        print("Example: python combine_rivals.py \"C:\\Users\\Saanit\\Downloads\\Linkoping FC\"")
        sys.exit(1)

    folder_path = sys.argv[1]
    combine_team_csvs(folder_path)


if __name__ == "__main__":
    main()
