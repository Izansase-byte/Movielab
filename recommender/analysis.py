from __future__ import annotations

import numpy as np
import pandas as pd


def movie_statistics(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Return one analysis-ready row per movie, including zero-rating titles."""
    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(vots="size", mitjana="mean", mediana="median", desviació="std", mínim="min", màxim="max")
        .reset_index()
    )
    result = movies[["movieId", "title", "genres"]].merge(stats, on="movieId", how="left")
    result["any"] = result["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).astype("Float64")
    result["vots"] = result["vots"].fillna(0).astype(int)
    return result


def user_statistics(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    stats = (
        ratings.groupby("userId")["rating"]
        .agg(valoracions="size", mitjana="mean", mediana="median", desviació="std", mínima="min", màxima="max")
        .reset_index()
    )
    genre_rows = ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    genre_rows = genre_rows.assign(gènere=genre_rows["genres"].fillna("").str.split("|")).explode("gènere")
    favorite = (
        genre_rows.groupby(["userId", "gènere"])
        .agg(n=("rating", "size"), nota=("rating", "mean"))
        .reset_index()
        .sort_values(["userId", "nota", "n"], ascending=[True, False, False])
        .drop_duplicates("userId")
        .rename(columns={"gènere": "gènere_preferit"})[["userId", "gènere_preferit"]]
    )
    return stats.merge(favorite, on="userId", how="left")


def genre_statistics(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    expanded = movies[["movieId", "genres"]].copy()
    expanded["gènere"] = expanded["genres"].fillna("Sense gènere").str.split("|")
    expanded = expanded.explode("gènere")
    movie_counts = expanded.groupby("gènere")["movieId"].nunique().rename("pel·lícules")
    rated = expanded.merge(ratings, on="movieId", how="left")
    rating_stats = rated.groupby("gènere").agg(
        valoracions=("rating", "count"),
        usuaris=("userId", "nunique"),
        mitjana=("rating", "mean"),
        desviació=("rating", "std"),
    )
    return movie_counts.to_frame().join(rating_stats).reset_index().sort_values("valoracions", ascending=False)


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values >= 0]
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    ordered = np.sort(values)
    n = len(ordered)
    return float((2 * np.sum(np.arange(1, n + 1) * ordered) / (n * ordered.sum())) - (n + 1) / n)


def data_quality_report(movies: pd.DataFrame, ratings: pd.DataFrame) -> dict[str, float | int | str]:
    movie_count = int(movies["movieId"].nunique())
    user_count = int(ratings["userId"].nunique())
    rating_count = int(len(ratings))
    item_counts = ratings["movieId"].value_counts()
    user_counts = ratings["userId"].value_counts()
    missing = int(movies[["title", "genres"]].isna().sum().sum() + ratings[["userId", "movieId", "rating"]].isna().sum().sum())
    duplicates = int(ratings.duplicated(["userId", "movieId"]).sum())
    report: dict[str, float | int | str] = {
        "Pel·lícules": movie_count,
        "Usuaris": user_count,
        "Valoracions": rating_count,
        "Densitat (%)": 100 * rating_count / max(1, movie_count * user_count),
        "Esparsitat (%)": 100 * (1 - rating_count / max(1, movie_count * user_count)),
        "Valors absents": missing,
        "Parelles duplicats": duplicates,
        "Usuaris amb < 5 vots": int((user_counts < 5).sum()),
        "Pel·lícules amb < 5 vots": int((item_counts < 5).sum()),
        "Pel·lícules sense vots": int(movie_count - item_counts.index.nunique()),
        "Vots mitjans per usuari": float(user_counts.mean()),
        "Vots mitjans per pel·lícula": float(item_counts.mean()),
        "Desigualtat de popularitat (Gini)": _gini(item_counts.to_numpy()),
        "Nota mínima": float(ratings["rating"].min()),
        "Nota màxima": float(ratings["rating"].max()),
        "Nota mitjana": float(ratings["rating"].mean()),
    }
    if "timestamp" in ratings and ratings["timestamp"].notna().any():
        dates = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce")
        if dates.notna().any():
            report["Primera valoració"] = str(dates.min().date())
            report["Última valoració"] = str(dates.max().date())
    return report


def popularity_curve(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Catalogue popularity and cumulative interaction share for long-tail analysis."""
    counts = ratings["movieId"].value_counts().rename("vots")
    result = movies[["movieId", "title"]].merge(counts, left_on="movieId", right_index=True, how="left")
    result["vots"] = result["vots"].fillna(0).astype(int)
    result = result.sort_values("vots", ascending=False).reset_index(drop=True)
    result["posició"] = np.arange(1, len(result) + 1)
    total = max(1, result["vots"].sum())
    result["percentatge_acumulat"] = 100 * result["vots"].cumsum() / total
    return result
