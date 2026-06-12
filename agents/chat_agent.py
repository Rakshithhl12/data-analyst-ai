"""Chat Agent — FAISS RAG + Gemini with improved answer quality."""
import json, logging, os
from pathlib import Path
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert data analyst assistant. You have access to a structured analysis of a dataset.

STRICT FORMATTING RULES — always follow these:
1. NEVER write paragraph answers. Every answer must use bullet points, numbered lists, or tables.
2. For statistics or comparisons → use a markdown table.
3. For findings, insights, recommendations → use numbered or bullet point lists.
4. For a single metric or fact → still use a bullet point.
5. Always use **bold** for column names, numbers, and key terms.
6. Start every answer with a short bold heading that matches the question (e.g. **Summary**, **Column Statistics**, **Top Values**).
7. Keep each bullet point concise — one idea per line.
8. If multiple columns are compared, always use a table with columns: | Column | Metric | Value |

EXAMPLE — good answer format:
**Column Statistics: Revenue**
| Metric | Value |
|--------|-------|
| Mean | 4,520 |
| Min | 100 |
| Max | 98,000 |
| Median | 3,200 |

**Key Observations:**
- **Revenue** is right-skewed — most values are low but a few are very high
- **25th percentile** is 800, suggesting majority of records are low-value
"""

class ChatAgent:
    TOP_K = 6

    def __init__(self):
        self._client = None
        self._has_llm = None

    def _init_llm(self):
        if self._has_llm is not None:
            return self._has_llm
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._has_llm = True
            logger.info("[ChatAgent] Gemini client initialized.")
        else:
            self._has_llm = False
            logger.warning("[ChatAgent] No GEMINI_API_KEY — keyword fallback.")
        return self._has_llm

    def answer(self, question: str, vector_store_path: str, session_id: str,
               history: List[dict] = None) -> str:
        self._init_llm()
        chunks = self._load_chunks(vector_store_path)
        if not chunks:
            return "⚠️ No analysis data found. Please run the full analysis first."

        ctx_chunks = self._retrieve(question, vector_store_path, chunks)
        context = "\n\n".join(ctx_chunks)

        if self._has_llm:
            return self._ask_gemini(question, context, history or [])
        return self._keyword_answer(ctx_chunks)

    def _load_chunks(self, vpath: str) -> List[str]:
        f = Path(vpath) / "chunks.json"
        if not f.exists():
            return []
        return json.loads(f.read_text())

    def _retrieve(self, query, vpath, chunks):
        # Filter out raw data row chunks — they hurt retrieval quality
        clean_chunks = [c for c in chunks if not c.startswith("Rows ")]
        faiss_idx = Path(vpath) / "index.faiss"
        if faiss_idx.exists():
            try:
                return self._faiss_search(query, str(faiss_idx), clean_chunks)
            except Exception as e:
                logger.warning("FAISS failed: %s", e)
        return self._keyword_search(query, clean_chunks)

    def _faiss_search(self, query, index_path, chunks):
        import faiss
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        qv = model.encode([query]).astype(np.float32)
        idx = faiss.read_index(index_path)
        k = min(self.TOP_K, len(chunks))
        _, indices = idx.search(qv, k)
        return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

    def _keyword_search(self, query, chunks):
        qw = set(query.lower().split())
        scored = [(len(qw & set(c.lower().split())), c) for c in chunks]
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:self.TOP_K]]

    def _ask_gemini(self, question: str, context: str, history: List[dict]) -> str:
        # Build conversation history for multi-turn awareness
        history_text = ""
        if history:
            recent = history[-6:]  # last 3 exchanges
            history_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in recent
            )
            history_text = f"\n\nCONVERSATION HISTORY (for context):\n{history_text}\n"

        prompt = (
            f"{SYSTEM_PROMPT}"
            f"{history_text}"
            f"\nDATASET CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {question}\n\n"
            f"Answer:"
        )
        try:
            resp = self._client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            return resp.text.strip()
        except Exception as e:
            logger.error("Gemini chat failed: %s", e)
            return f"❌ AI model error: {e}. Please check your GEMINI_API_KEY."

    def _keyword_answer(self, chunks: List[str]) -> str:
        if not chunks:
            return "No relevant information found in the dataset analysis."
        lines = []
        for c in chunks[:4]:
            lines.append(f"- {c[:300]}")
        return "**Based on the dataset analysis:**\n\n" + "\n\n".join(lines)