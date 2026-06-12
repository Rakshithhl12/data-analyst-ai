"""
LangGraph Workflow — Fixed & Production Ready
"""
import json, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from agents.data_loader import DataLoaderAgent
from agents.cleaner import DataCleanerAgent
from agents.eda import EDAAgent
from agents.visualizer import VisualizerAgent
from agents.insights import InsightsAgent
from agents.reporter import ReporterAgent

logger = logging.getLogger(__name__)

class AnalysisState(TypedDict):
    filepath: str
    session_id: str
    charts_dir: str
    reports_dir: str
    vector_store_dir: str
    df_json: Optional[str]
    understanding: Optional[Dict[str, Any]]
    cleaning_report: Optional[Dict[str, Any]]
    eda_results: Optional[Dict[str, Any]]
    chart_paths: Optional[List[str]]
    insights: Optional[Dict[str, Any]]
    report_paths: Optional[Dict[str, str]]
    error: Optional[str]
    status: Optional[str]

def _node_load(s):
    logger.info("[load] %s", s["filepath"])
    try:
        r = DataLoaderAgent().run(filepath=s["filepath"])
        s["df_json"] = r["df_json"]
        s["understanding"] = r["understanding"]
        s["status"] = "loaded"
    except Exception as e:
        logger.exception("[load] FAILED"); s["error"] = str(e); s["status"] = "failed"
    return s

def _node_clean(s):
    if s.get("error"): return s
    logger.info("[clean]")
    try:
        r = DataCleanerAgent().run(df_json=s["df_json"])
        s["df_json"] = r["df_json"]; s["cleaning_report"] = r["cleaning_report"]; s["status"] = "cleaned"
    except Exception as e:
        logger.exception("[clean] FAILED"); s["error"] = str(e); s["status"] = "failed"
    return s

def _node_eda(s):
    if s.get("error"): return s
    logger.info("[eda]")
    try:
        s["eda_results"] = EDAAgent().run(df_json=s["df_json"]); s["status"] = "eda_done"
    except Exception as e:
        logger.exception("[eda] FAILED"); s["error"] = str(e); s["status"] = "failed"
    return s

def _node_visualize(s):
    if s.get("error"): return s
    logger.info("[visualize]")
    try:
        Path(s["charts_dir"]).mkdir(parents=True, exist_ok=True)
        r = VisualizerAgent().run(df_json=s["df_json"], eda_results=s["eda_results"], charts_dir=s["charts_dir"])
        s["chart_paths"] = r["chart_paths"]; s["status"] = "visualized"
    except Exception as e:
        logger.exception("[visualize] FAILED"); s["error"] = str(e); s["status"] = "failed"; s["chart_paths"] = []
    return s

def _node_insights(s):
    if s.get("error"): return s
    logger.info("[insights]")
    try:
        s["insights"] = InsightsAgent().run(
            understanding=s["understanding"],
            cleaning_report=s["cleaning_report"],
            eda_results=s["eda_results"])
        s["status"] = "insights_done"
    except Exception as e:
        logger.exception("[insights] FAILED"); s["error"] = str(e); s["status"] = "failed"
    return s

def _node_report(s):
    if s.get("error"): return s
    logger.info("[report]")
    try:
        Path(s["reports_dir"]).mkdir(parents=True, exist_ok=True)
        Path(s["vector_store_dir"]).mkdir(parents=True, exist_ok=True)
        r = ReporterAgent().run(
            session_id=s["session_id"], understanding=s["understanding"],
            cleaning_report=s["cleaning_report"], eda_results=s["eda_results"],
            insights=s["insights"], chart_paths=s["chart_paths"] or [],
            charts_dir=s["charts_dir"], reports_dir=s["reports_dir"],
            df_json=s["df_json"], vector_store_dir=s["vector_store_dir"])
        s["report_paths"] = r; s["status"] = "complete"
    except Exception as e:
        logger.exception("[report] FAILED"); s["error"] = str(e); s["status"] = "failed"
    return s

def _build():
    g = StateGraph(AnalysisState)
    for name, fn in [("load",_node_load),("clean",_node_clean),("eda",_node_eda),
                     ("visualize",_node_visualize),("insights",_node_insights),("report",_node_report)]:
        g.add_node(name, fn)
    g.set_entry_point("load")
    g.add_edge("load","clean"); g.add_edge("clean","eda"); g.add_edge("eda","visualize")
    g.add_edge("visualize","insights"); g.add_edge("insights","report"); g.add_edge("report",END)
    return g.compile()

_WF = _build()

def run_analysis_workflow(filepath, session_id, charts_dir, reports_dir, vector_store_dir):
    init: AnalysisState = {
        "filepath":filepath,"session_id":session_id,"charts_dir":charts_dir,
        "reports_dir":reports_dir,"vector_store_dir":vector_store_dir,
        "df_json":None,"understanding":None,"cleaning_report":None,"eda_results":None,
        "chart_paths":None,"insights":None,"report_paths":None,"error":None,"status":"starting"
    }
    final = _WF.invoke(init)
    summary = {
        "session_id":session_id, "status":final.get("status","unknown"),
        "error":final.get("error"), "understanding":final.get("understanding"),
        "cleaning_report":final.get("cleaning_report"), "eda_results":final.get("eda_results"),
        "insights":final.get("insights"), "chart_paths":final.get("chart_paths") or [],
        "report_paths":final.get("report_paths") or {}
    }
    p = Path(reports_dir) / "status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Workflow done. status=%s", summary["status"])
    return summary
