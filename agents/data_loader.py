"""Data Loader Agent — reads CSV/Excel, detects schema."""
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DataLoaderAgent:
    def run(self, filepath: str) -> Dict[str, Any]:
        df = self._load(filepath)
        understanding = self._understand(df, Path(filepath).name)
        df_json = df.to_json(orient="split", date_format="iso", default_handler=str)
        return {"df_json": df_json, "understanding": understanding}

    def _load(self, filepath: str) -> pd.DataFrame:
        ext = Path(filepath).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(filepath, low_memory=False)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file: {ext}")
        logger.info("Loaded %s rows × %s cols", *df.shape)
        return df

    def _understand(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        rows, cols = df.shape
        info = {}
        for col in df.columns:
            s = df[col]
            nc = int(s.isna().sum())
            ci = {"dtype": str(s.dtype), "null_count": nc,
                  "null_pct": round(nc/rows*100, 2) if rows > 0 else 0,
                  "unique_count": int(s.nunique(dropna=True))}
            if pd.api.types.is_numeric_dtype(s):
                d = s.describe()
                ci.update({"min": _sv(d.get("min")), "max": _sv(d.get("max")),
                            "mean": _sv(d.get("mean")), "std": _sv(d.get("std"))})
            else:
                ci["top_values"] = {str(k): int(v) for k,v in s.value_counts().head(5).items()}
            info[col] = ci
        return {
            "filename": filename,
            "shape": {"rows": rows, "cols": cols},
            "columns": list(df.columns),
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object","category"]).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=["datetime","datetimetz"]).columns.tolist(),
            "total_missing": int(df.isna().sum().sum()),
            "total_missing_pct": round(df.isna().sum().sum()/(rows*cols)*100, 2) if rows*cols > 0 else 0,
            "duplicate_rows": int(df.duplicated().sum()),
            "columns_info": info,
            "memory_usage_kb": round(df.memory_usage(deep=True).sum()/1024, 2)
        }

def _sv(v):
    if v is None: return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v
