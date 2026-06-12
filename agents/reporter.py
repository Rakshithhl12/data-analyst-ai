"""
Reporter Agent — Fixed PDF (ReportLab), Full HTML, FAISS store
==============================================================
Uses ReportLab for PDF (no system dependencies like WeasyPrint).
HTML report is fully self-contained with base64 charts.
"""
import base64, json, logging, os
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_CIRC = 238.76  # 2π × 38

class ReporterAgent:
    def run(self, session_id, understanding, cleaning_report, eda_results,
            insights, chart_paths, charts_dir, reports_dir, df_json, vector_store_dir):
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        html = self._html(session_id, understanding, cleaning_report, eda_results, insights, chart_paths)
        html_path = Path(reports_dir) / "report.html"
        html_path.write_text(html, encoding="utf-8")
        logger.info("[Reporter] HTML: %s", html_path)

        pdf_path = Path(reports_dir) / "report.pdf"
        self._pdf(understanding, cleaning_report, eda_results, insights, chart_paths, str(pdf_path))
        logger.info("[Reporter] PDF: %s", pdf_path)

        df = pd.read_json(StringIO(df_json), orient="split")
        self._build_vs(df, understanding, eda_results, insights, vector_store_dir)

        return {"html": str(html_path), "pdf": str(pdf_path)}

    # ── HTML ─────────────────────────────────────────────────────────────────
    def _html(self, session_id, u, c, e, ins, chart_paths):
        shape    = u.get("shape", {})
        fname    = u.get("filename", "dataset")
        num_cols = u.get("numeric_columns", [])
        cat_cols = u.get("categorical_columns", [])
        missing  = u.get("total_missing", 0)
        miss_pct = u.get("total_missing_pct", 0.0)
        dupes    = u.get("duplicate_rows", 0)
        ins      = ins or {}
        quality  = int(ins.get("data_quality_score", 80))
        q_color  = "#22c55e" if quality>=80 else "#f59e0b" if quality>=60 else "#ef4444"
        q_dash   = round(_CIRC * quality / 100, 2)
        q_label  = "Excellent" if quality>=90 else "Good" if quality>=80 else "Fair" if quality>=60 else "Poor"

        # Charts base64
        chart_html = ""
        for p in chart_paths:
            path = Path(p)
            if path.suffix == ".png" and path.exists():
                b64 = base64.b64encode(path.read_bytes()).decode()
                title = path.stem.replace("_"," ").title()
                chart_html += (f'<div class="chart-card"><p class="ct">{title}</p>'
                               f'<img src="data:image/png;base64,{b64}" alt="{title}"/></div>')
        if not chart_html:
            chart_html = '<p class="dim" style="grid-column:1/-1;text-align:center">No charts generated.</p>'

        # Cleaning
        actions_html = ""
        for a in (c or {}).get("actions", []):
            actions_html += (f'<div class="action-row"><span class="ba">'
                             f'{a.get("action","").replace("_"," ").title()}</span>'
                             f'<span class="dim-txt">{a.get("description","")}</span></div>')
        if not actions_html:
            actions_html = '<p class="dim">No cleaning actions required.</p>'

        # Stats table
        stats_body = "".join(
            f"<tr><td class='cn'>{col}</td><td>{_f(s.get('mean'))}</td><td>{_f(s.get('std'))}</td>"
            f"<td>{_f(s.get('min'))}</td><td>{_f(s.get('50%'))}</td><td>{_f(s.get('max'))}</td>"
            f"<td>{int(s.get('count',0)):,}</td></tr>"
            for col,s in (e or {}).get("summary_stats",{}).items()
        )
        stats_section = (
            "<div class='tw'><table class='dt'><thead><tr>"
            "<th>Column</th><th>Mean</th><th>Std</th><th>Min</th><th>Median</th><th>Max</th><th>Count</th>"
            f"</tr></thead><tbody>{stats_body}</tbody></table></div>"
        ) if stats_body else '<p class="dim">No numeric columns.</p>'

        # Correlation
        pairs = (e or {}).get("correlation",{}).get("high_correlation_pairs",[])
        corr_body = "".join(
            f"<tr><td class='cn'>{p['col_a']}</td><td class='cn'>{p['col_b']}</td>"
            f"<td class='{'pos' if p['correlation']>0 else 'neg'}'>{p['correlation']:+.3f}</td>"
            f"<td>{_cs(p['correlation'])}</td></tr>" for p in pairs)
        corr_section = (
            "<div class='tw'><table class='dt'><thead><tr><th>Column A</th><th>Column B</th>"
            f"<th>r</th><th>Strength</th></tr></thead><tbody>{corr_body}</tbody></table></div>"
        ) if corr_body else '<p class="dim">No strong correlations (|r| ≥ 0.5).</p>'

        # Distribution
        dist_body = "".join(
            f"<tr><td class='cn'>{col}</td>"
            f"<td class='{'pos' if d['skewness']>=0 else 'neg'}'>{d['skewness']:+.3f}</td>"
            f"<td>{d['kurtosis']:+.3f}</td><td style='font-size:.82rem'>{d.get('skew_interpretation','')}</td>"
            f"<td>{_f(d.get('percentiles',{}).get('p25'))}</td>"
            f"<td>{_f(d.get('percentiles',{}).get('p75'))}</td></tr>"
            for col,d in (e or {}).get("distribution_analysis",{}).items()
        )
        dist_section = (
            "<div class='tw'><table class='dt'><thead><tr>"
            "<th>Column</th><th>Skewness</th><th>Kurtosis</th><th>Shape</th><th>P25</th><th>P75</th>"
            f"</tr></thead><tbody>{dist_body}</tbody></table></div>"
        ) if dist_body else '<p class="dim">No numeric columns.</p>'

        # Top-N bars
        top_n_html = ""
        for key, agg in list((e or {}).get("top_n_analysis",{}).items())[:3]:
            items = [(k,v) for k,v in list(agg.items())[:8] if v is not None]
            if not items: continue
            max_v = max(v for _,v in items) or 1
            bars = "".join(
                f'<div class="bar-row"><span class="bl">{str(k)[:20]}</span>'
                f'<div class="bt"><div class="bf" style="width:{min(100,v/max_v*100):.1f}%"></div></div>'
                f'<span class="bv">{_f(v,0)}</span></div>' for k,v in items)
            top_n_html += f'<div class="topn"><p class="tn-h">{key.replace("_vs_"," → ")}</p>{bars}</div>'
        if not top_n_html:
            top_n_html = '<p class="dim">No aggregation data.</p>'

        # Column profiles
        col_rows = "".join(
            f"<tr><td class='cn'>{col}</td><td><span class='dtype'>{info.get('dtype','?')}</span></td>"
            f"<td>{info.get('null_count',0):,} <span class='dim'>({info.get('null_pct',0):.1f}%)</span>"
            f"<div class='nt'><div class='nb' style='width:{min(100,info.get('null_pct',0)):.0f}%'></div></div></td>"
            f"<td>{info.get('unique_count',0):,}</td></tr>"
            for col,info in list(u.get("columns_info",{}).items())[:40]
        )

        def _li(items): return "".join(f"<li>{x}</li>" for x in items) or "<li>None detected.</li>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Analysis Report — {fname}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f0f23;color:#e2e8f0;line-height:1.65}}
a{{color:#06b6d4;text-decoration:none}}
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:#070714}}
::-webkit-scrollbar-thumb{{background:#334155;border-radius:99px}}
.wrap{{max-width:1180px;margin:0 auto;padding:2rem 1.5rem}}
header{{background:linear-gradient(135deg,#312e81 0%,#4f46e5 50%,#0891b2 100%);padding:3rem 2.5rem;
  border-radius:16px;margin-bottom:2.5rem;position:relative;overflow:hidden}}
header::before{{content:'';position:absolute;inset:0;
  background:repeating-linear-gradient(45deg,rgba(255,255,255,.03) 0,rgba(255,255,255,.03) 1px,transparent 1px,transparent 12px)}}
header h1{{font-size:2rem;font-weight:800;color:#fff;margin-bottom:.3rem;position:relative}}
.h-sub{{color:rgba(255,255,255,.75);font-size:.93rem;position:relative}}
.h-meta{{display:flex;gap:.8rem;margin-top:1rem;flex-wrap:wrap;position:relative}}
.h-chip{{background:rgba(255,255,255,.13);border-radius:8px;padding:.3rem .8rem;font-size:.78rem;color:#fff}}
.sg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.9rem;margin-bottom:1.4rem}}
.sc{{background:#1a1a2e;border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:1.1rem;text-align:center}}
.sc-ico{{font-size:1.3rem;margin-bottom:.2rem}}
.sc-val{{font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#4f46e5,#06b6d4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.sc-lbl{{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.07em;margin-top:.15rem}}
.card{{background:#1a1a2e;border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:1.4rem;margin-bottom:1.2rem}}
.card-t{{font-size:.82rem;font-weight:700;color:#a5b4fc;text-transform:uppercase;letter-spacing:.07em;
  margin-bottom:1rem;display:flex;align-items:center;gap:.5rem}}
.card-t::after{{content:'';flex:1;height:1px;background:rgba(255,255,255,.08)}}
.qw{{display:flex;align-items:center;gap:2rem;flex-wrap:wrap}}
.qc{{position:relative;width:86px;height:86px;flex-shrink:0}}
.qc svg{{transform:rotate(-90deg)}}
.qs{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-size:1.3rem;font-weight:800;color:{q_color}}}
.qd h4{{font-size:1.05rem;font-weight:700;color:{q_color}}}
.qd p{{font-size:.84rem;color:#94a3b8;margin-top:.3rem;line-height:1.6}}
.action-row{{display:flex;align-items:flex-start;gap:.7rem;padding:.6rem .85rem;
  background:#0f172a;border-left:3px solid #4f46e5;border-radius:0 8px 8px 0;margin-bottom:.5rem}}
.ba{{background:rgba(79,70,229,.2);color:#a5b4fc;border-radius:999px;padding:.15rem .6rem;
  font-size:.73rem;font-weight:700;white-space:nowrap;flex-shrink:0}}
.dim-txt{{font-size:.875rem;color:#94a3b8}}
.tw{{overflow-x:auto}}
.dt{{width:100%;border-collapse:collapse;font-size:.84rem}}
.dt th{{background:rgba(79,70,229,.22);color:#a5b4fc;padding:.56rem .8rem;text-align:left;
  font-weight:600;font-size:.76rem;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}}
.dt td{{padding:.46rem .8rem;border-bottom:1px solid rgba(255,255,255,.04);color:#94a3b8}}
.dt tr:hover td{{background:rgba(255,255,255,.02)}}
.cn{{color:#e2e8f0;font-weight:500}}.pos{{color:#4ade80}}.neg{{color:#f87171}}
.dtype{{background:rgba(6,182,212,.12);color:#22d3ee;border-radius:4px;padding:.07rem .4rem;
  font-size:.77rem;font-family:monospace}}
.nt{{height:4px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden;margin-top:.2rem;width:70px}}
.nb{{height:100%;background:#ef4444;border-radius:99px}}
.topn{{margin-bottom:1.4rem}}
.tn-h{{font-size:.8rem;font-weight:700;color:#a5b4fc;margin-bottom:.65rem;text-transform:uppercase;letter-spacing:.05em}}
.bar-row{{display:flex;align-items:center;gap:.5rem;margin-bottom:.35rem}}
.bl{{width:115px;font-size:.79rem;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0}}
.bt{{flex:1;height:9px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden}}
.bf{{height:100%;background:linear-gradient(90deg,#4f46e5,#06b6d4);border-radius:99px}}
.bv{{width:60px;text-align:right;font-size:.79rem;color:#e2e8f0;font-weight:600;flex-shrink:0}}
.ig{{display:grid;grid-template-columns:repeat(auto-fit,minmax(265px,1fr));gap:.9rem;margin-bottom:1.2rem}}
.ic{{background:#0f172a;border-radius:10px;padding:1.1rem;border-left:3px solid #4f46e5}}
.ic.g{{border-left-color:#22c55e}}.ic.a{{border-left-color:#f59e0b}}
.ic.r{{border-left-color:#ef4444}}.ic.c{{border-left-color:#06b6d4}}
.ic h5{{font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.5rem;color:#a5b4fc}}
.ic.g h5{{color:#22c55e}}.ic.a h5{{color:#f59e0b}}.ic.r h5{{color:#ef4444}}.ic.c h5{{color:#06b6d4}}
.ic li{{font-size:.85rem;color:#94a3b8;line-height:1.75;margin-left:1rem}}
.exec{{background:linear-gradient(135deg,rgba(79,70,229,.08),rgba(6,182,212,.04));
  border:1px solid rgba(79,70,229,.35);border-radius:10px;padding:1.3rem 1.5rem;
  line-height:1.9;color:#94a3b8;font-size:.93rem}}
.cg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1.1rem}}
.chart-card{{background:#0f172a;border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:1rem}}
.chart-card:hover{{border-color:rgba(79,70,229,.4)}}
.ct{{font-size:.77rem;color:#64748b;margin-bottom:.5rem;font-weight:500}}
.chart-card img{{width:100%;border-radius:6px;display:block}}
.print-btn{{display:inline-flex;align-items:center;gap:.4rem;
  background:linear-gradient(135deg,#4f46e5,#7c3aed);border:none;color:#fff;
  padding:.55rem 1.2rem;border-radius:8px;cursor:pointer;font-size:.875rem;font-weight:600}}
footer{{text-align:center;padding:2rem 0;color:#64748b;font-size:.8rem;
  border-top:1px solid rgba(255,255,255,.08);margin-top:2.5rem}}
.dim{{color:#64748b}}
@media print{{
  body{{background:#fff!important;color:#000!important}}
  .card{{background:#fff!important;border:1px solid #e2e8f0!important}}
  header{{background:#4f46e5!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
  .print-btn{{display:none!important}}
  .dt td,.dt th{{color:#000!important;border-color:#e2e8f0!important}}
  .cn{{color:#1e293b!important}}.dim{{color:#64748b!important}}
  .sc-val{{color:#4f46e5!important;-webkit-text-fill-color:#4f46e5!important}}
}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>📊 Data Analysis Report</h1>
  <p class="h-sub">{fname}</p>
  <div class="h-meta">
    <span class="h-chip">🗂 Session {session_id[:8]}…</span>
    <span class="h-chip">📐 {shape.get('rows',0):,} rows × {shape.get('cols',0)} cols</span>
    <span class="h-chip">⚡ LangGraph + Gemini</span>
  </div>
</header>

<div class="sg">
  <div class="sc"><div class="sc-ico">📋</div><div class="sc-val">{shape.get('rows',0):,}</div><div class="sc-lbl">Total Rows</div></div>
  <div class="sc"><div class="sc-ico">🏛</div><div class="sc-val">{shape.get('cols',0)}</div><div class="sc-lbl">Columns</div></div>
  <div class="sc"><div class="sc-ico">🔢</div><div class="sc-val">{len(num_cols)}</div><div class="sc-lbl">Numeric Cols</div></div>
  <div class="sc"><div class="sc-ico">🔤</div><div class="sc-val">{len(cat_cols)}</div><div class="sc-lbl">Categorical</div></div>
  <div class="sc"><div class="sc-ico">❓</div><div class="sc-val">{missing:,}</div><div class="sc-lbl">Missing ({miss_pct:.1f}%)</div></div>
  <div class="sc"><div class="sc-ico">🗑</div><div class="sc-val">{dupes:,}</div><div class="sc-lbl">Duplicates</div></div>
</div>

<div class="card">
  <div class="card-t">📐 Data Quality Score</div>
  <div class="qw">
    <div class="qc">
      <svg width="86" height="86" viewBox="0 0 86 86">
        <circle cx="43" cy="43" r="38" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="7"/>
        <circle cx="43" cy="43" r="38" fill="none" stroke="{q_color}" stroke-width="7"
          stroke-dasharray="{q_dash} {_CIRC}" stroke-linecap="round"/>
      </svg>
      <div class="qs">{quality}</div>
    </div>
    <div class="qd">
      <h4>{q_label} Quality</h4>
      <p>Based on {miss_pct:.1f}% missing data and {dupes} duplicate rows.<br/>
      Score: <strong style="color:{q_color}">{quality}/100</strong></p>
    </div>
  </div>
</div>

<div class="card">
  <div class="card-t">🗂 Column Profiles</div>
  <div class="tw"><table class="dt">
    <thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique Values</th></tr></thead>
    <tbody>{col_rows}</tbody>
  </table></div>
</div>

<div class="card">
  <div class="card-t">🧹 Data Cleaning Summary</div>
  {actions_html}
</div>

<div class="card"><div class="card-t">📈 Summary Statistics</div>{stats_section}</div>
<div class="card"><div class="card-t">🔗 Correlation Analysis</div>{corr_section}</div>
<div class="card"><div class="card-t">📊 Distribution Analysis</div>{dist_section}</div>
<div class="card"><div class="card-t">🏆 Top-N Aggregations</div>{top_n_html}</div>

<div class="card">
  <div class="card-t">🤖 Executive Summary</div>
  <div class="exec">{ins.get('executive_summary','Analysis complete.')}</div>
</div>

<div class="ig">
  <div class="ic g"><h5>✅ Key Findings</h5><ul>{_li(ins.get('key_findings',[]))}</ul></div>
  <div class="ic c"><h5>📈 Trends</h5><ul>{_li(ins.get('trends',[]))}</ul></div>
  <div class="ic a"><h5>⚠ Anomalies</h5><ul>{_li(ins.get('anomalies',[]))}</ul></div>
  <div class="ic"  ><h5>💡 Recommendations</h5><ul>{_li(ins.get('business_recommendations',[]))}</ul></div>
  <div class="ic r"><h5>🚨 Risk Factors</h5><ul>{_li(ins.get('risk_factors',[]))}</ul></div>
</div>

<div class="card">
  <div class="card-t">📉 Visualisations</div>
  <div class="cg">{chart_html}</div>
</div>

<footer>
  Autonomous Data Analyst &nbsp;·&nbsp; LangGraph + Gemini + FAISS
  &nbsp;&nbsp;
  <button class="print-btn" onclick="window.print()">🖨 Print / Save as PDF</button>
</footer>
</div>
</body></html>"""

    # ── PDF with ReportLab ────────────────────────────────────────────────────
    def _pdf(self, understanding, cleaning_report, eda_results, insights, chart_paths, pdf_path):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                            Table, TableStyle, HRFlowable, Image as RLImage)
            from reportlab.lib.enums import TA_CENTER, TA_LEFT

            doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                    leftMargin=2*cm, rightMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            story  = []

            # Custom styles
            h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20,
                                 textColor=colors.HexColor("#4f46e5"), spaceAfter=6)
            h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                                 textColor=colors.HexColor("#7c3aed"), spaceAfter=4, spaceBefore=12)
            body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9,
                                  textColor=colors.HexColor("#374151"), leading=14)
            dim  = ParagraphStyle("Dim", parent=styles["Normal"], fontSize=8,
                                  textColor=colors.HexColor("#6b7280"))

            shape = understanding.get("shape", {})
            fname = understanding.get("filename", "dataset")
            ins   = insights or {}

            # Title
            story.append(Paragraph("📊 Data Analysis Report", h1))
            story.append(Paragraph(f"File: {fname}  |  Rows: {shape.get('rows',0):,}  |  Cols: {shape.get('cols',0)}", dim))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")))
            story.append(Spacer(1, 0.3*cm))

            # Quality
            quality = int(ins.get("data_quality_score", 80))
            story.append(Paragraph(f"Data Quality Score: {quality}/100", h2))
            story.append(Paragraph(ins.get("executive_summary","Analysis complete."), body))
            story.append(Spacer(1, 0.3*cm))

            # Overview table
            story.append(Paragraph("Dataset Overview", h2))
            num_c = understanding.get("numeric_columns",[])
            cat_c = understanding.get("categorical_columns",[])
            ov_data = [
                ["Metric", "Value"],
                ["Total Rows",      f"{shape.get('rows',0):,}"],
                ["Total Columns",   str(shape.get('cols',0))],
                ["Numeric Columns", str(len(num_c))],
                ["Categorical Cols",str(len(cat_c))],
                ["Missing Values",  f"{understanding.get('total_missing',0):,} ({understanding.get('total_missing_pct',0):.1f}%)"],
                ["Duplicate Rows",  str(understanding.get('duplicate_rows',0))],
            ]
            ov_tbl = Table(ov_data, colWidths=[6*cm, 10*cm])
            ov_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4f46e5")),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                ("FONTSIZE",   (0,0), (-1,-1), 9),
                ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#f9fafb")),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f3f4f6")]),
                ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("RIGHTPADDING",(0,0), (-1,-1), 8),
                ("TOPPADDING",  (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ]))
            story.append(ov_tbl)
            story.append(Spacer(1, 0.3*cm))

            # Key Findings
            story.append(Paragraph("Key Findings", h2))
            for kf in ins.get("key_findings", []):
                story.append(Paragraph(f"• {kf}", body))
            story.append(Spacer(1, 0.2*cm))

            # Recommendations
            story.append(Paragraph("Business Recommendations", h2))
            for r in ins.get("business_recommendations", []):
                story.append(Paragraph(f"• {r}", body))
            story.append(Spacer(1, 0.2*cm))

            # Summary stats
            ss = (eda_results or {}).get("summary_stats", {})
            if ss:
                story.append(Paragraph("Summary Statistics", h2))
                cols_header = ["Column", "Mean", "Std", "Min", "Median", "Max"]
                ss_data = [cols_header] + [
                    [col, _f(s.get("mean")), _f(s.get("std")),
                     _f(s.get("min")), _f(s.get("50%")), _f(s.get("max"))]
                    for col,s in list(ss.items())[:15]
                ]
                ss_tbl = Table(ss_data, colWidths=[4*cm,2.5*cm,2.5*cm,2.5*cm,2.5*cm,2.5*cm])
                ss_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#7c3aed")),
                    ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                    ("FONTSIZE",   (0,0), (-1,-1), 8),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f3f4f6")]),
                    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
                    ("LEFTPADDING", (0,0), (-1,-1), 5),
                    ("RIGHTPADDING",(0,0), (-1,-1), 5),
                    ("TOPPADDING",  (0,0), (-1,-1), 3),
                    ("BOTTOMPADDING",(0,0),(-1,-1), 3),
                ]))
                story.append(ss_tbl)
                story.append(Spacer(1, 0.3*cm))

            # Charts
            png_charts = [p for p in chart_paths if p.endswith(".png") and Path(p).exists()]
            if png_charts:
                story.append(Paragraph("Visualisations", h2))
                story.append(Spacer(1, 0.2*cm))
                for cp in png_charts[:12]:
                    try:
                        img = RLImage(cp, width=14*cm, height=8*cm, kind="proportional")
                        story.append(img)
                        story.append(Paragraph(Path(cp).stem.replace("_"," ").title(), dim))
                        story.append(Spacer(1, 0.3*cm))
                    except Exception as e:
                        logger.warning("Could not embed chart %s: %s", cp, e)

            doc.build(story)
            logger.info("[Reporter] ReportLab PDF done: %s", pdf_path)

        except ImportError:
            logger.warning("reportlab not installed — writing placeholder PDF")
            Path(pdf_path).write_bytes(b"%PDF-1.4\n%%EOF")
        except Exception as e:
            logger.exception("PDF generation failed: %s", e)
            Path(pdf_path).write_bytes(b"%PDF-1.4\n%%EOF")

    # ── FAISS vector store ────────────────────────────────────────────────────
    def _build_vs(self, df, understanding, eda_results, insights, vdir):
        Path(vdir).mkdir(parents=True, exist_ok=True)
        chunks = self._chunks(df, understanding, eda_results, insights)
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(chunks, show_progress_bar=False).astype(np.float32)
            idx = faiss.IndexFlatL2(emb.shape[1])
            idx.add(emb)
            faiss.write_index(idx, str(Path(vdir)/"index.faiss"))
            logger.info("[Reporter] FAISS index built (%d chunks)", len(chunks))
        except Exception as e:
            logger.warning("FAISS skipped: %s", e)
        with open(Path(vdir)/"chunks.json","w") as f:
            json.dump(chunks, f, indent=2)

    def _chunks(self, df, u, e, ins):
        ch = []
        s = u.get("shape", {})
        rows, cols = s.get("rows", 0), s.get("cols", 0)
        num_cols = u.get("numeric_columns", [])
        cat_cols = u.get("categorical_columns", [])
        missing  = u.get("total_missing", 0)
        miss_pct = u.get("total_missing_pct", 0.0)
        dupes    = u.get("duplicate_rows", 0)

        # 1. Dataset overview
        ch.append(
            f"Dataset overview: The file '{u.get('filename', 'dataset')}' has {rows:,} rows and {cols} columns. "
            f"It contains {len(num_cols)} numeric columns ({', '.join(num_cols)}) and "
            f"{len(cat_cols)} categorical columns ({', '.join(cat_cols)}). "
            f"Total missing values: {missing} ({miss_pct:.1f}%). Duplicate rows found: {dupes}."
        )

        # 2. Numeric column stats — full natural language sentences
        for col, st in (e or {}).get("summary_stats", {}).items():
            ch.append(
                f"Column '{col}' statistics: average (mean) is {_f(st.get('mean'))}, "
                f"standard deviation is {_f(st.get('std'))}, "
                f"minimum value is {_f(st.get('min'))}, maximum value is {_f(st.get('max'))}, "
                f"median (50th percentile) is {_f(st.get('50%'))}, "
                f"25th percentile is {_f(st.get('25%'))}, 75th percentile is {_f(st.get('75%'))}, "
                f"count of non-null values is {int(st.get('count', 0)):,}."
            )

        # 3. Distribution / skew info
        for col, info in (e or {}).get("distribution_analysis", {}).items():
            skew = info.get("skewness")
            interp = info.get("skew_interpretation", "")
            kurt = info.get("kurtosis")
            if skew is not None:
                ch.append(
                    f"Distribution of '{col}': skewness={_f(skew)} ({interp}), "
                    f"kurtosis={_f(kurt)}. "
                    + ("This column is heavily skewed and may need transformation before modelling."
                       if "highly" in interp else "")
                )

        # 4. Categorical columns — top values
        for col, info in (e or {}).get("categorical_analysis", {}).items():
            unique = info.get("unique_count", "?")
            top = ", ".join(f"{k} ({v} times)" for k, v in list(info.get("top_20", {}).items())[:8])
            ch.append(
                f"Categorical column '{col}' has {unique} unique values. "
                f"Most frequent values: {top}."
            )

        # 5. Correlations
        hc = (e or {}).get("correlation", {}).get("high_correlation_pairs", [])
        if hc:
            pairs = "; ".join(
                f"'{p['col_a']}' and '{p['col_b']}' (r={p['correlation']}, {_cs(p['correlation'])})"
                for p in hc[:10]
            )
            ch.append(f"High correlation pairs in the dataset: {pairs}.")

        # 6. Top-N aggregations
        for key, agg in (e or {}).get("top_n_analysis", {}).items():
            entries = ", ".join(f"{k} = {v}" for k, v in list(agg.items())[:8])
            ch.append(f"Aggregation — {key}: {entries}.")

        # 7. Insights — executive summary
        exec_sum = (ins or {}).get("executive_summary", "")
        if exec_sum:
            ch.append(f"Executive summary: {exec_sum}")

        # 8. Key findings
        for x in (ins or {}).get("key_findings", []):
            ch.append(f"Key finding: {x}")

        # 9. Trends
        for x in (ins or {}).get("trends", []):
            ch.append(f"Trend observed: {x}")

        # 10. Anomalies
        for x in (ins or {}).get("anomalies", []):
            ch.append(f"Anomaly: {x}")

        # 11. Business recommendations
        for x in (ins or {}).get("business_recommendations", []):
            ch.append(f"Business recommendation: {x}")

        # 12. Risk factors
        for x in (ins or {}).get("risk_factors", []):
            ch.append(f"Risk factor: {x}")

        # 13. Data quality score
        score = (ins or {}).get("data_quality_score")
        if score is not None:
            label = "Excellent" if score >= 90 else "Good" if score >= 80 else "Fair" if score >= 60 else "Poor"
            ch.append(f"Data quality score: {score}/100 ({label}).")

        # NOTE: Raw row dumps intentionally excluded — they hurt retrieval quality.
        return ch

def _f(v, d=2):
    if v is None: return "—"
    try: return f"{float(v):,.{d}f}"
    except: return str(v)

def _cs(r):
    a = abs(r)
    if a>=.9: return "Very Strong"
    if a>=.7: return "Strong"
    if a>=.5: return "Moderate"
    return "Weak"