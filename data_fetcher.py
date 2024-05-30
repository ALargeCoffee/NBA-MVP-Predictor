import pandas as pd

# Fetches and cleans data
def fetch_data():

    # Read the file and select appropriate columns by index
    data = pd.read_csv('./data/NBA_Player_Stats.csv')
    selected_cols = [1, 3, 5, 7, 10, 13, 16, 17, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    trimmed_data = data.iloc[:, selected_cols]

    # Clean dupes and invalid entries from each season
    is_duped = trimmed_data.duplicated(['Player', 'Season'])
    no_dupes = trimmed_data[~is_duped]
    cleaned_data = no_dupes.dropna()
    return cleaned_data