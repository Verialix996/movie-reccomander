import os
import sys
import zipfile
import requests
import pandas as pd

RAW_DIR = "data/raw"
OUT_PATH = "data/movies.csv"

SOURCES = {
    "imdb_basics": "https://datasets.imdbws.com/title.basics.tsv.gz",
    "imdb_ratings": "https://datasets.imdbws.com/title.ratings.tsv.gz",
    "movielens": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
}


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  already exists, skipping: {dest}")
        return
    print(f"  downloading {url}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp_dest = f"{dest}.part"
    if os.path.exists(tmp_dest):
        os.remove(tmp_dest)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"    {pct:.1f}%", end="\r", flush=True)
    os.replace(tmp_dest, dest)
    print(f"    done ({downloaded / 1e6:.1f} MB)")


def step_download() -> bool:
    print("\n[1/4] Downloading raw datasets...")
    download(SOURCES["imdb_basics"], f"{RAW_DIR}/title.basics.tsv.gz")
    download(SOURCES["imdb_ratings"], f"{RAW_DIR}/title.ratings.tsv.gz")

    ml_zip = f"{RAW_DIR}/ml-latest.zip"
    ml_dir = f"{RAW_DIR}/ml-latest"
    try:
        download(SOURCES["movielens"], ml_zip)
        if not os.path.exists(ml_dir):
            print("  extracting ml-latest.zip ...")
            with zipfile.ZipFile(ml_zip, "r") as z:
                z.extractall(RAW_DIR)
            print("  extracted.")
        return True
    except (requests.RequestException, zipfile.BadZipFile, OSError) as e:
        print(f"  MovieLens unavailable, continuing with IMDb-only data: {e}")
        return False


def step_imdb() -> pd.DataFrame:
    print("\n[2/4] Processing IMDb data...")
    print("  reading title.basics (this may take a minute)...")
    basics = pd.read_csv(
        f"{RAW_DIR}/title.basics.tsv.gz",
        sep="\t",
        na_values=r"\N",
        low_memory=False,
        usecols=["tconst", "titleType", "primaryTitle", "startYear", "isAdult", "genres"],
    )
    movies = basics[
        (basics["titleType"] == "movie")
        & (basics["isAdult"] == 0)
        & basics["genres"].notna()
        & basics["startYear"].notna()
    ].copy()
    movies.drop(columns=["titleType", "isAdult"], inplace=True)
    movies.rename(columns={"primaryTitle": "title", "startYear": "year"}, inplace=True)
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    movies.dropna(subset=["year"], inplace=True)
    movies["year"] = movies["year"].astype(int)
    print(f"  {len(movies):,} movies after type/adult filter")

    print("  reading title.ratings...")
    ratings = pd.read_csv(
        f"{RAW_DIR}/title.ratings.tsv.gz",
        sep="\t",
        na_values=r"\N",
    )
    merged = movies.merge(ratings, on="tconst", how="inner")
    merged = merged[merged["numVotes"] >= 100]
    merged.rename(columns={"averageRating": "imdb_rating", "numVotes": "imdb_votes"}, inplace=True)
    print(f"  {len(merged):,} movies after rating/vote filter (numVotes >= 100)")
    return merged[["tconst", "title", "year", "genres", "imdb_rating", "imdb_votes"]]


def step_movielens() -> pd.DataFrame:
    print("\n[3/4] Processing MovieLens data...")
    ml_dir = f"{RAW_DIR}/ml-latest"

    print("  computing per-movie average ratings...")
    ratings = pd.read_csv(f"{ml_dir}/ratings.csv", usecols=["movieId", "rating"])
    agg = ratings.groupby("movieId")["rating"].agg(ml_rating="mean", ml_votes="count").reset_index()
    agg["ml_rating"] = (agg["ml_rating"] * 2).round(2)  # normalize 0-5 → 0-10

    links = pd.read_csv(f"{ml_dir}/links.csv", usecols=["movieId", "imdbId"])
    links.dropna(subset=["imdbId"], inplace=True)
    links["imdbId"] = links["imdbId"].astype(int)
    links["tconst"] = links["imdbId"].apply(lambda x: f"tt{x:07d}")

    ml = agg.merge(links[["movieId", "tconst"]], on="movieId", how="inner")
    print(f"  {len(ml):,} MovieLens movies with IMDb links")
    return ml[["tconst", "ml_rating", "ml_votes"]]


def empty_movielens() -> pd.DataFrame:
    print("\n[3/4] Skipping MovieLens data...")
    return pd.DataFrame(columns=["tconst", "ml_rating", "ml_votes"])


def step_merge(imdb_df: pd.DataFrame, ml_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/4] Merging and saving...")
    combined = imdb_df.merge(ml_df, on="tconst", how="left")
    combined.rename(columns={"tconst": "imdb_id"}, inplace=True)
    combined.drop_duplicates(subset=["imdb_id"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    os.makedirs("data", exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"  saved {len(combined):,} movies → {OUT_PATH}")
    print("\nSample:")
    print(combined[["title", "year", "genres", "imdb_rating", "imdb_votes"]].head(5).to_string(index=False))
    print(f"\nRating distribution:\n{combined['imdb_rating'].describe().to_string()}")
    return combined


if __name__ == "__main__":
    if os.path.exists(OUT_PATH):
        resp = input(f"{OUT_PATH} already exists. Re-build? [y/N] ").strip().lower()
        if resp != "y":
            print("Skipping. Delete data/movies.csv to force a rebuild.")
            sys.exit(0)

    has_movielens = step_download()
    imdb_df = step_imdb()
    ml_df = step_movielens() if has_movielens else empty_movielens()
    step_merge(imdb_df, ml_df)
    print("\nDone! Run: streamlit run app.py")
