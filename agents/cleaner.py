"""Data Cleaner Agent — imputation, dedup, outlier detection."""
import logging
from io import StringIO
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DataCleanerAgent:
    def run(self, df_json: str) -> Dict[str, Any]:
        df = pd.read_json(StringIO(df_json), orient="split")
        report: Dict[str, Any] = {
            "original_shape": {"rows": df.shape[0], "cols": df.shape[1]},
            "actions": []
        }
        df, report = self._dedup(df, report)
        df, report = self._impute(df, report)
        df, report = self._outliers(df, report)
        df, report = self._fix_dtypes(df, report)
        report["final_shape"] = {"rows": df.shape[0], "cols": df.shape[1]}
        report["remaining_missing"] = int(df.isna().sum().sum())
        return {"df_json": df.to_json(orient="split", date_format="iso", default_handler=str),
                "cleaning_report": report}

    def _dedup(self, df, report):
        n = int(df.duplicated().sum())
        if n:
            df = df.drop_duplicates().reset_index(drop=True)
            report["actions"].append({"action":"remove_duplicates","rows_removed":n,
                "description":f"Removed {n} duplicate rows."})
        return df, report

    def _impute(self, df, report):
        imputed = []
        for col in df.columns:
            n = int(df[col].isna().sum())
            if not n: continue
            if pd.api.types.is_numeric_dtype(df[col]):
                fv = df[col].median()
                df[col] = df[col].fillna(fv)
                imputed.append({"column":col,"strategy":"median","fill_value":_sv(fv),"cells_filled":n})
            else:
                mv = df[col].mode()
                fv = mv[0] if len(mv) else "Unknown"
                df[col] = df[col].fillna(fv)
                imputed.append({"column":col,"strategy":"mode","fill_value":str(fv),"cells_filled":n})
        if imputed:
            report["actions"].append({"action":"impute_missing","columns_affected":len(imputed),
                "details":imputed,"description":f"Imputed missing values in {len(imputed)} column(s)."})
        return df, report

    def _outliers(self, df, report):
        summary = []
        for col in df.select_dtypes(include="number").columns:
            q1,q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3-q1
            if iqr == 0: continue
            mask = (df[col] < q1-1.5*iqr)|(df[col] > q3+1.5*iqr)
            n = int(mask.sum())
            if n: summary.append({"column":col,"outlier_count":n,
                "lower_bound":_sv(q1-1.5*iqr),"upper_bound":_sv(q3+1.5*iqr)})
        if summary:
            report["actions"].append({"action":"detect_outliers","method":"IQR (1.5×)",
                "columns":summary,"description":"Outliers detected via IQR. Flagged but not removed."})
            report["outlier_details"] = summary
        return df, report

    def _fix_dtypes(self, df, report):
        conversions = []
        for col in df.select_dtypes(include="object").columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum()/max(len(df),1) > 0.8:
                df[col] = converted
                conversions.append({"column":col,"from":"object","to":"numeric"})
                continue
            try:
                cdt = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if cdt.notna().sum()/max(len(df),1) > 0.8:
                    df[col] = cdt
                    conversions.append({"column":col,"from":"object","to":"datetime"})
            except Exception: pass
        if conversions:
            report["actions"].append({"action":"fix_dtypes","conversions":conversions,
                "description":f"Auto-converted {len(conversions)} column(s)."})
        return df, report

def _sv(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v
