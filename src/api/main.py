"""FastAPI エンドポイント。セッションスコープのパイプライン管理。

[Codex adversarial-review fix: high]
旧実装: グローバル単一パイプラインを全APIコーラーが共有 → テナント分離違反。
修正: セッションID必須。各セッションが独立したパイプラインを持つ。
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from src.api.session import ConversationSession, SessionManager
from src.pipeline import PipelineResult
from src.schemas.memory import RetrievalQuery

app = FastAPI(
    title="Emotion-Capable AI API",
    description="感情を明示的内部状態として持つ研究用AIシステムのAPI（セッションスコープ）",
    version="0.2.0",
)

_manager = SessionManager()


def _get_session(session_id: str) -> ConversationSession:
    """セッションIDからセッションを取得。存在しなければ404。"""
    session = _manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. POST /sessions to create.")
    return session


# --- リクエスト/レスポンスモデル ---

class TextInput(BaseModel):
    text: str
    event_type: str = "user_message"


class StateResponse(BaseModel):
    session_id: str
    state: dict
    step_count: int
    regulation_mode: str


class MemoryQueryRequest(BaseModel):
    query_text: str = ""
    query_tags: list[str] = Field(default_factory=list)
    max_results: int = 10


# --- セッション管理 ---

@app.post("/sessions")
def create_session():
    """新しいセッションを作成する。"""
    session = _manager.create_session()
    return {"session_id": session.session_id, "created_at": session.created_at.isoformat()}


@app.get("/sessions")
def list_sessions():
    """全セッション一覧を返す。"""
    return {"sessions": [s.model_dump() for s in _manager.list_sessions()]}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """セッションを削除する。"""
    if not _manager.remove_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


# --- セッションスコープのエンドポイント ---

@app.post("/sessions/{session_id}/process")
def process_text(session_id: str, input_data: TextInput):
    """テキスト入力をセッションの情動パイプラインで処理する。"""
    session = _get_session(session_id)
    return session.process(input_data.text)


@app.get("/sessions/{session_id}/state", response_model=StateResponse)
def get_current_state(session_id: str):
    """セッションの現在の情動状態を取得する。"""
    session = _get_session(session_id)
    state = session.pipeline.current_state
    return StateResponse(
        session_id=session_id,
        state={name: getattr(state, name) for name in state.variable_names()},
        step_count=state.step_count,
        regulation_mode=state.regulation_mode,
    )


@app.post("/sessions/{session_id}/tick")
def tick(session_id: str):
    """セッションの内部時計を進める。"""
    session = _get_session(session_id)
    session.pipeline.tick()
    return {"status": "ok", "step_count": session.pipeline.current_state.step_count}


@app.post("/sessions/{session_id}/reset")
def reset(session_id: str):
    """セッションをリセットする。"""
    session = _get_session(session_id)
    session.reset()
    return {"status": "reset", "step_count": 0}


@app.post("/sessions/{session_id}/memory/query")
def query_memory(session_id: str, request: MemoryQueryRequest):
    """セッションの記憶を検索する。"""
    session = _get_session(session_id)
    state = session.pipeline.current_state
    query = RetrievalQuery(
        query_text=request.query_text,
        query_tags=request.query_tags,
        current_valence=state.valence,
        current_arousal=state.arousal,
        max_results=request.max_results,
    )
    results = session.pipeline.memory_store.retrieve(query)
    return {
        "session_id": session_id,
        "count": len(results),
        "results": [
            {
                "summary": r.memory.summary,
                "salience": r.memory.emotional_salience,
                "relevance": r.relevance_score,
                "combined": r.combined_score,
            }
            for r in results
        ],
    }


@app.get("/sessions/{session_id}/audit/recent")
def recent_audit(session_id: str, limit: int = 20):
    """セッションの直近の監査ログを取得する。"""
    session = _get_session(session_id)
    buffer = session.pipeline.audit_logger.get_buffer()
    return {"session_id": session_id, "entries": buffer[-limit:]}


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}
