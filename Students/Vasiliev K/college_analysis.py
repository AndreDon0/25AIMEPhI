
import os
import sys
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sns.set(style="whitegrid", rc={"figure.figsize": (8, 5)})


def try_numeric(s: pd.Series) -> pd.Series:
    """Вычищаем всё лишнее"""
    return pd.to_numeric(s.astype(str).str.replace(",", "").replace("", "nan"), errors="coerce")


def pearson_corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    return df.corr(method="pearson").iloc[0, 1]


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    return spearmanr(df.iloc[:, 0], df.iloc[:, 1]).correlation


def regplot_show_or_save(df: pd.DataFrame, x: str, y: str, show: bool, outpath: Optional[str] = None):
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x=x, y=y, scatter_kws={"s": 25}, line_kws={"color": "red"}, ci=None)
    p = pearson_corr(df[x], df[y])
    s = spearman_corr(df[x], df[y])
    txt = f"Pearson: {p:.3f}\nSpearman: {s:.3f}"
    plt.gca().text(0.02, 0.95, txt, transform=plt.gca().transAxes,
                   fontsize=9, va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    plt.title(f"{x} vs {y}")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        if outpath:
            plt.savefig(outpath, dpi=200)
        plt.close()


def boxplot_show_or_save(df: pd.DataFrame, cat: str, val: str, show: bool, outpath: Optional[str] = None):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=cat, y=val)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{val} by {cat}")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        if outpath:
            plt.savefig(outpath, dpi=200)
        plt.close()


def heatmap_show_or_save(df: pd.DataFrame, cols: List[str], show: bool, outpath: Optional[str] = None):
    if len(cols) < 2:
        return
    plt.figure(figsize=(10, 8))
    corr = df[cols].corr(method="pearson")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Pearson correlation matrix")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        if outpath:
            plt.savefig(outpath, dpi=200)
        plt.close()


def pairplot_show_or_save(df: pd.DataFrame, cols: List[str], show: bool, outpath: Optional[str] = None):
    if len(cols) < 2:
        return
    cols = cols[:6]
    sub = df[cols].dropna()
    if sub.shape[0] < 2:
        return
    g = sns.pairplot(sub, diag_kind="kde", plot_kws={"s": 20})
    g.fig.suptitle("Pairplot (selected numeric columns)", y=1.02)
    if show:
        plt.show()
    else:
        if outpath:
            g.fig.tight_layout()
            g.fig.savefig(outpath, dpi=200)
        plt.close(g.fig)


def pick_columns(df: pd.DataFrame):
    # Лучшие предположения в наших колонках
    lower = {c.lower(): c for c in df.columns}
    def find(cands):
        for c in cands:
            if c.lower() in lower:
                return lower[c.lower()]
        # substring search as fallback
        for c in df.columns:
            for cand in cands:
                if cand.lower() in c.lower():
                    return c
        return None

    ug = find(["ug_fee", "ug fee", "ugfee", "ug_fee", "ug_fee"])
    pg = find(["pg_fee", "pg fee", "pgfee", "pg_fee"])
    stream = find(["stream", "major", "department", "dept"])
    rating = find(["rating", "score"])
    placement = find(["placement", "placements"])
    faculty = find(["faculty"])
    academic = find(["academic"])
    infra = find(["infrastructure", "infrastructure"])
    social = find(["social", "social_life", "social life", "social_life"])
    return {
        "UG_fee": ug,
        "PG_fee": pg,
        "Stream": stream,
        "Rating": rating,
        "Placement": placement,
        "Faculty": faculty,
        "Academic": academic,
        "Infrastructure": infra,
        "Social_Life": social
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal college visualizer (only numpy/pandas/matplotlib/seaborn/scipy).")
    parser.add_argument("--input", "-i", required=False, default="College_data.csv", help="Path to CSV (default: College_data.csv)")
    parser.add_argument("--save", "-s", required=False, default=None, help="If given, directory to save PNGs (no display).")
    args = parser.parse_args()

    infile = args.input
    outdir = args.save
    show_plots = outdir is None

    if not os.path.exists(infile):
        print("Input file not found:", infile)
        sys.exit(1)

    df = pd.read_csv(infile)
    print("Loaded", infile, "shape", df.shape)
    print("Columns:", list(df.columns))

    cols_map = pick_columns(df)
    print("Detected mapping:", cols_map)

    # Try to coerce fee columns and numeric-looking cols to numeric
    for k in ("UG_fee", "PG_fee"):
        c = cols_map.get(k)
        if c and c in df.columns:
            df[c] = try_numeric(df[c])

    # Also coerce a few others if present
    for maybe in ("Rating", "Placement", "Faculty", "Academic", "Infrastructure", "Social_Life"):
        c = cols_map.get(maybe)
        if c and c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = try_numeric(df[c])

    tasks = []

    # UG_fee vs Rating
    if cols_map["UG_fee"] and cols_map["Rating"]:
        tasks.append(("reg", cols_map["UG_fee"], cols_map["Rating"], f"UGfee_vs_Rating.png"))
    else:
        print("Skipping UG_fee vs Rating")

    # UG_fee vs Placement
    if cols_map["UG_fee"] and cols_map["Placement"]:
        tasks.append(("reg", cols_map["UG_fee"], cols_map["Placement"], "UGfee_vs_Placement.png"))
    else:
        print("Skipping UG_fee vs Placement")

    # Faculty vs Rating
    if cols_map["Faculty"] and cols_map["Rating"]:
        tasks.append(("reg", cols_map["Faculty"], cols_map["Rating"], "Faculty_vs_Rating.png"))
    else:
        print("Skipping Faculty vs Rating")

    # Boxplots by Stream for fees/rating/placement
    stream_col = cols_map["Stream"]
    if stream_col:
        for candidate in ("UG_fee", "PG_fee", "Rating", "Placement"):
            c = cols_map.get(candidate)
            if c and c in df.columns:
                tasks.append(("box", stream_col, c, f"{stream_col}_by_{c}.png"))
    else:
        print("No Stream column detected — skipping category boxplots")

    # Prepare numeric columns list for heatmap/pairplot
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Prefer relevant numeric ones (intersection)
    preferred = [v for v in cols_map.values() if v and v in numeric_cols]
    # Add any other numeric columns up to 6
    final_numeric = list(dict.fromkeys(preferred + numeric_cols))[:6]

    if len(final_numeric) >= 2:
        tasks.append(("heat", final_numeric, None, "correlation_heatmap.png"))
        tasks.append(("pair", final_numeric, None, "pairplot.png"))
    else:
        print("Not enough numeric columns for heatmap/pairplot:", final_numeric)

    # Execute tasks: show or save
    if show_plots:
        print("Displaying plots (interactive). Close figure windows to continue.")
    else:
        os.makedirs(outdir, exist_ok=True)
        print("Saving plots to:", outdir)

    for kind, a, b, fname in tasks:
        try:
            if kind == "reg":
                outpath = os.path.join(outdir, fname) if outdir else None
                regplot_show_or_save(df, a, b, show_plots, outpath)
            elif kind == "box":
                outpath = os.path.join(outdir, fname) if outdir else None
                boxplot_show_or_save(df, a, b, show_plots, outpath)
            elif kind == "heat":
                outpath = os.path.join(outdir, fname) if outdir else None
                heatmap_show_or_save(df, a, show_plots, outpath)
            elif kind == "pair":
                outpath = os.path.join(outdir, fname) if outdir else None
                pairplot_show_or_save(df, a, show_plots, outpath)
        except Exception as e:
            print("Error on task", kind, a, b, ":", e)

    print("Finished.")

if __name__ == "__main__":
    main()
