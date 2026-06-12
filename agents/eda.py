"""EDA Agent — stats, correlation, distributions, top-N."""
import logging
from io import StringIO
from typing import Any, Dict, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class EDAAgent:
    def run(self, df_json: str) -> Dict[str, Any]:
        df = pd.read_json(StringIO(df_json), orient="split")
        logger.info("[EDA] shape=%s", df.shape)
        return {
            "summary_stats": self._stats(df),
            "correlation": self._corr(df),
            "categorical_analysis": self._cat(df),
            "distribution_analysis": self._dist(df),
            "top_n_analysis": self._topn(df),
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object","category"]).columns.tolist(),
            "shape": {"rows": df.shape[0], "cols": df.shape[1]}
        }

    def _stats(self, df):
        num = df.select_dtypes(include="number")
        if num.empty: return {}
        desc = num.describe().round(4)
        return {col: {k: _sv(v) for k,v in desc[col].items()} for col in desc.columns}

    def _corr(self, df):
        num = df.select_dtypes(include="number")
        if num.shape[1] < 2: return {}
        corr = num.corr().round(4)
        matrix = {col: {row: (None if np.isnan(v) else float(v))
                        for row,v in corr[col].items()} for col in corr.columns}
        cols = corr.columns.tolist()
        pairs = []
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                v = corr.iloc[i,j]
                if not np.isnan(v) and abs(v) >= 0.5:
                    pairs.append({"col_a":cols[i],"col_b":cols[j],"correlation":round(float(v),4)})
        return {"matrix": matrix, "high_correlation_pairs": pairs}

    def _cat(self, df):
        result = {}
        for col in df.select_dtypes(include=["object","category"]).columns:
            vc = df[col].value_counts(dropna=False).head(20)
            result[col] = {"unique_count": int(df[col].nunique(dropna=True)),
                           "top_20": {str(k): int(v) for k,v in vc.items()}}
        return result

    def _dist(self, df):
        dist = {}
        for col in df.select_dtypes(include="number").columns:
            s = df[col].dropna()
            if s.empty: continue
            sk = float(s.skew()); ku = float(s.kurtosis())
            dist[col] = {"skewness": round(sk,4), "kurtosis": round(ku,4),
                "skew_interpretation": _skew(sk),
                "percentiles": {"p5":_sv(s.quantile(.05)),"p25":_sv(s.quantile(.25)),
                    "p50":_sv(s.quantile(.5)),"p75":_sv(s.quantile(.75)),"p95":_sv(s.quantile(.95))}}
        return dist

    def _topn(self, df, n=10):
        cat = df.select_dtypes(include=["object","category"]).columns.tolist()
        num = df.select_dtypes(include="number").columns.tolist()
        result = {}
        for c in cat[:3]:
            if df[c].nunique() > 100: continue
            for nc in num[:3]:
                key = f"{c}_vs_{nc}"
                grp = df.groupby(c)[nc].sum().sort_values(ascending=False).head(n)
                result[key] = {str(k): _sv(v) for k,v in grp.items()}
        return result

def _sv(v):
    if v is None: return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v

def _skew(s):
    if abs(s) < 0.5: return "approximately symmetric"
    return "positively skewed (right tail)" if s > 0 else "negatively skewed (left tail)"
