import pandas as pd

def normalize_genres(genres_str):

    genres = sorted(set(genres_str.split('|')))
    return '|'.join(genres)

    df_combined['genres'] = df_combined['genres'].apply(normalize_genres)
    df_separate_aggregated['genres'] = df_separate_aggregated['genres'].apply(normalize_genres)

    df_merged = pd.merge(
        df_combined,
        df_separate_aggregated,
        on=['userId', 'movieId', 'rating', 'timestamp', 'title', 'genres'],
        how='outer')
    return df_merged

def preprocess_genres(df_combined, df_separate):

    df_separate_aggregated = df_separate.groupby([
        'userId', 'movieId', 'rating', 'timestamp', 'title'
    ])['genres'].apply('|'.join).reset_index()

    # Deduplicate to avoid duplicates
    df_separate_aggregated = df_separate_aggregated.drop_duplicates()

    # Normalize genres
    df_combined['genres'] = df_combined['genres'].apply(normalize_genres)
    df_separate_aggregated['genres'] = df_separate_aggregated['genres'].apply(normalize_genres)

    # Merge DataFrames
    df_merged = pd.merge(
        df_combined,
        df_separate_aggregated,
        on=['userId', 'movieId', 'rating', 'timestamp', 'title', 'genres'],
        how='outer'
    )

    return df_merged


