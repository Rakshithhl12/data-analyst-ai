"""Insights Agent — Gemini-powered analysis with rule-based fallback."""
import json, logging, os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class InsightsAgent:
    MODEL = "gemini-1.5-flash"

    def __init__(self):
        # Lazy-init: don't read env vars at import time.
        self._client = None
        self._has_key = None  # None = not yet checked

    def _init_llm(self):
        """Lazy-initialize Gemini client on first use."""
        if self._has_key is not None:
            return self._has_key
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._has_key = True
            logger.info("[InsightsAgent] Gemini client initialized.")
        else:
            logger.warning("[InsightsAgent] GEMINI_API_KEY not set — using rule-based fallback.")
            self._has_key = False
        return self._has_key

    def run(self, understanding, cleaning_report, eda_results):
        self._init_llm()
        context = self._context(understanding, cleaning_report, eda_results)
        if self._has_key:
            try:
                ins = self._gemini(context)
                logger.info("[Insights] Gemini OK")
                return ins
            except Exception as e:
                logger.warning("[Insights] Gemini failed: %s", e)
        return self._fallback(understanding, cleaning_report, eda_results)

    def _context(self, u, c, e):
        shape = u.get("shape", {})
        num   = u.get("numeric_columns", [])
        cat   = u.get("categorical_columns", [])
        hc    = e.get("correlation", {}).get("high_correlation_pairs", [])
        dist  = e.get("distribution_analysis", {})
        parts = [
            f"Dataset: {u.get('filename','unknown')}",
            f"Shape: {shape.get('rows','?')} rows x {shape.get('cols','?')} columns",
            f"Numeric columns ({len(num)}): {', '.join(num[:10])}",
            f"Categorical columns ({len(cat)}): {', '.join(cat[:10])}",
            f"Total missing: {u.get('total_missing',0)}",
            f"Duplicate rows: {u.get('duplicate_rows',0)}",
        ]
        if hc:
            parts.append("High correlations: " + "; ".join(
                f"{p['col_a']} & {p['col_b']} (r={p['correlation']})" for p in hc[:5]))
        skewed = [f"{col} ({info['skew_interpretation']})" for col,info in dist.items()
                  if "skewed" in info.get("skew_interpretation","")]
        if skewed: parts.append(f"Skewed: {', '.join(skewed[:5])}")
        topn = e.get("top_n_analysis", {})
        if topn:
            fk = next(iter(topn))
            parts.append(f"Top ({fk}): " + ", ".join(f"{k}={v}" for k,v in list(topn[fk].items())[:3]))
        return "\n".join(parts)

    def _gemini(self, context):
        prompt = (
            "You are a senior data analyst. Based on the dataset summary below, provide a professional analysis.\n\n"
            f"DATASET SUMMARY:\n{context}\n\n"
            "Respond ONLY with valid JSON (no markdown, no code fences) with this exact structure:\n"
            '{"executive_summary":"2-3 paragraph summary","key_findings":["finding 1","finding 2","finding 3","finding 4","finding 5"],'
            '"trends":["trend 1","trend 2","trend 3"],"anomalies":["anomaly 1","anomaly 2"],'
            '"business_recommendations":["rec 1","rec 2","rec 3","rec 4"],"risk_factors":["risk 1","risk 2"],'
            '"data_quality_score":85}'
        )
        resp = self._client.models.generate_content(model=self.MODEL, contents=prompt)
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw.strip())

    def _fallback(self, u, c, e):
        shape = u.get("shape", {}); rows = shape.get("rows", 0); cols = shape.get("cols", 0)
        num = u.get("numeric_columns", []); cat = u.get("categorical_columns", [])
        mp  = u.get("total_missing_pct", 0); dupes = u.get("duplicate_rows", 0)
        hc  = e.get("correlation", {}).get("high_correlation_pairs", [])
        quality = 100
        if mp > 20: quality -= 20
        elif mp > 5: quality -= 10
        if dupes > 0: quality -= 5
        findings = [
            f"Dataset contains {rows:,} rows and {cols} columns.",
            f"{len(num)} numeric and {len(cat)} categorical columns detected.",
            f"Overall missing data: {mp:.1f}%.",
        ]
        if dupes: findings.append(f"{dupes} duplicate rows identified and removed.")
        if hc:
            p = hc[0]
            findings.append(f"Strong correlation (r={p['correlation']}) between '{p['col_a']}' and '{p['col_b']}'.")
        trends = []
        for col, info in list(e.get("distribution_analysis",{}).items())[:3]:
            lbl = info.get("skew_interpretation","")
            if lbl != "approximately symmetric": trends.append(f"'{col}' is {lbl}.")
        return {
            "executive_summary": (
                f"The dataset '{u.get('filename','dataset')}' consists of {rows:,} records and {cols} attributes. "
                f"Data quality is {'good' if quality>=80 else 'moderate'} with {mp:.1f}% missing values. "
                f"Analysis identified {len(num)} numeric and {len(cat)} categorical dimensions."
            ),
            "key_findings": findings,
            "trends": trends or ["No strong distributional trends detected."],
            "anomalies": ["Outliers detected — review IQR analysis in the cleaning report."
                          if c.get("outlier_details") else "No critical anomalies flagged."],
            "business_recommendations": [
                "Address remaining missing values before model training.",
                "Explore highly correlated variable pairs for feature engineering.",
                "Segment dataset by categorical dimensions for deeper insights.",
                "Normalise skewed numeric columns before statistical modelling.",
            ],
            "risk_factors": [
                "Low data volume may limit statistical power." if rows < 500 else "Sufficient data volume.",
                "High-cardinality categorical columns may need encoding.",
            ],
            "data_quality_score": quality,
        }