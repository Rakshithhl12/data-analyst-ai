"""Visualizer Agent — generates all chart types as PNG and Plotly HTML."""
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

class VisualizerAgent:
    def run(self, df_json: str, eda_results: Dict[str, Any], charts_dir: str) -> Dict[str, List[str]]:
        df = pd.read_json(StringIO(df_json), orient="split")
        out = Path(charts_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        num = eda_results.get("numeric_columns", [])
        cat = eda_results.get("categorical_columns", [])

        for c in num[:6]:    paths += [p for p in [self._hist(df,c,out)] if p]
        for c in cat[:4]:    paths += [p for p in [self._bar(df,c,out)] if p]
        for c in cat[:2]:    paths += [p for p in [self._pie(df,c,out)] if p]
        if len(num) >= 2:    paths += [p for p in [self._scatter(df,num[0],num[1],out)] if p]
        if len(num) >= 4:    paths += [p for p in [self._scatter(df,num[2],num[3],out)] if p]
        if len(num) >= 2:    paths += [p for p in [self._heatmap(df,num,out)] if p]
        for c in num[:4]:    paths += [p for p in [self._box(df,c,out)] if p]
        if num: paths += [p for p in [self._plotly_hist(df,num[0],out)] if p]
        if cat and num: paths += [p for p in [self._plotly_bar(df,cat[0],num[0],out)] if p]

        logger.info("[Visualizer] Generated %d charts", len(paths))
        return {"chart_paths": paths}

    def _hist(self, df, col, out):
        try:
            fig, ax = plt.subplots(figsize=(7,4))
            df[col].dropna().plot.hist(bins=30, ax=ax, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.set_title(f"Distribution of {col}", fontsize=12, fontweight="bold"); ax.set_xlabel(col); ax.set_ylabel("Frequency")
            fname = f"hist_{_slug(col)}.png"; fig.tight_layout(); fig.savefig(out/fname, bbox_inches="tight"); plt.close(fig)
            return str(out/fname)
        except Exception as e: logger.warning("Hist %s: %s",col,e); plt.close("all"); return None

    def _bar(self, df, col, out):
        try:
            vc = df[col].value_counts().head(15)
            if vc.empty: return None
            fig, ax = plt.subplots(figsize=(8,4))
            vc.plot.bar(ax=ax, color="#55A868", edgecolor="white")
            ax.set_title(f"Top Values — {col}", fontsize=12, fontweight="bold"); ax.tick_params(axis="x",rotation=45)
            fname = f"bar_{_slug(col)}.png"; fig.tight_layout(); fig.savefig(out/fname, bbox_inches="tight"); plt.close(fig)
            return str(out/fname)
        except Exception as e: logger.warning("Bar %s: %s",col,e); plt.close("all"); return None

    def _pie(self, df, col, out):
        try:
            vc = df[col].value_counts().head(8)
            if len(vc) < 2: return None
            fig, ax = plt.subplots(figsize=(6,6))
            vc.plot.pie(ax=ax, autopct="%1.1f%%", startangle=140, pctdistance=0.82)
            ax.set_ylabel(""); ax.set_title(f"Share — {col}", fontsize=12, fontweight="bold")
            fname = f"pie_{_slug(col)}.png"; fig.tight_layout(); fig.savefig(out/fname, bbox_inches="tight"); plt.close(fig)
            return str(out/fname)
        except Exception as e: logger.warning("Pie %s: %s",col,e); plt.close("all"); return None

    def _scatter(self, df, x, y, out):
        try:
            fig, ax = plt.subplots(figsize=(7,5))
            ax.scatter(df[x].dropna(), df[y].dropna(), alpha=0.4, color="#C44E52", s=15)
            ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"Scatter: {x} vs {y}", fontsize=12, fontweight="bold")
            fname = f"scatter_{_slug(x)}_vs_{_slug(y)}.png"; fig.tight_layout(); fig.savefig(out/fname, bbox_inches="tight"); plt.close(fig)
            return str(out/fname)
        except Exception as e: logger.warning("Scatter: %s",e); plt.close("all"); return None

    def _heatmap(self, df, num_cols, out):
        try:
            corr = df[num_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sz = max(6, len(num_cols))
            fig, ax = plt.subplots(figsize=(sz, sz-1))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                        vmin=-1, vmax=1, ax=ax, linewidths=0.5)
            ax.set_title("Correlation Heatmap", fontsize=12, fontweight="bold")
            fname = "correlation_heatmap.png"; fig.tight_layout(); fig.savefig(out/fname, bbox_inches="tight"); plt.close(fig)
            return str(out/fname)
        except Exception as e: logger.warning("Heatmap: %s",e); plt.close("all"); return None

    def _box(self, df, col, out):
        try:
            fig, ax = plt.subplots(figsize=(5,4))
            df[[col]].dropna().boxplot(ax=ax, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#8172B2",color="white"),
                medianprops=dict(color="white",linewidth=2))
            ax.set_title(f"Box Plot — {col}", fontsize=12, fontweight="bold")
            fname = f"box_{_slug(col)}.png"; fig.tight_layout(); fig.savefig(out/fname, bbox_inches="tight"); plt.close(fig)
            return str(out/fname)
        except Exception as e: logger.warning("Box %s: %s",col,e); plt.close("all"); return None

    def _plotly_hist(self, df, col, out):
        try:
            fig = px.histogram(df, x=col, nbins=40, title=f"Interactive Distribution — {col}", template="plotly_white")
            fname = f"plotly_hist_{_slug(col)}.html"
            fig.write_html(str(out/fname), include_plotlyjs="cdn"); return str(out/fname)
        except Exception as e: logger.warning("Plotly hist: %s",e); return None

    def _plotly_bar(self, df, cat_col, num_col, out):
        try:
            if df[cat_col].nunique() > 30: return None
            grp = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(15)
            fig = px.bar(x=grp.index.astype(str), y=grp.values,
                         labels={"x":cat_col,"y":num_col},
                         title=f"Top {cat_col} by {num_col}", template="plotly_white",
                         color=grp.values, color_continuous_scale="Blues")
            fname = f"plotly_bar_{_slug(cat_col)}.html"
            fig.write_html(str(out/fname), include_plotlyjs="cdn"); return str(out/fname)
        except Exception as e: logger.warning("Plotly bar: %s",e); return None

def _slug(n): return "".join(c if c.isalnum() else "_" for c in n)[:40]
