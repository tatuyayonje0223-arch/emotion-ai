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


# === EmotionBrain エンドポイント（正式システム） ===
# [NC1修正] APIからEmotionBrainに直接アクセス可能にする

_brains: dict[str, "EmotionBrain"] = {}


@app.post("/brain/create")
def create_brain():
    """EmotionBrain(正式最終システム)のセッションを作成する。"""
    from src.brian2_circuits.integrated_brain import EmotionBrain
    brain = EmotionBrain()
    from uuid import uuid4
    brain_id = f"brain-{uuid4().hex[:8]}"
    _brains[brain_id] = brain
    return {"brain_id": brain_id}


@app.post("/brain/{brain_id}/process")
def brain_process(brain_id: str, input_data: TextInput):
    """EmotionBrainでテキストを処理する。"""
    if brain_id not in _brains:
        raise HTTPException(404, f"Brain '{brain_id}' not found")
    result = _brains[brain_id].process(input_data.text)
    return {
        "brain_id": brain_id,
        "step": result.step,
        "blocked": result.blocked,
        "readout": result.readout.model_dump(),
        "text": input_data.text,
        "policy": result.policy.model_dump(),
        "neuromodulation": result.neuromodulation,
        "theta_coherence": result.theta_coherence,
        "memory_stats": result.memory_stats,
        "virtual_neurons": result.virtual_neurons,
        "region_activities": result.region_activities,
    }


@app.post("/brain/{brain_id}/sleep")
def brain_sleep(brain_id: str, cycles: int = 1):
    """EmotionBrainの睡眠リプレイを実行する。"""
    if brain_id not in _brains:
        raise HTTPException(404, f"Brain '{brain_id}' not found")
    results = _brains[brain_id].sleep(n_cycles=cycles)
    return {"brain_id": brain_id, "sleep_results": results}


# === IntegratedBrainV2 エンドポイント（10情動 + AdEx対応） ===

_brains_v2: dict[str, "IntegratedBrainV2"] = {}


class BrainV2CreateRequest(BaseModel):
    use_adex: bool = False


class BrainV2ProcessRequest(BaseModel):
    text: str
    context: float = 0.0


@app.post("/brain/v2/create")
def create_brain_v2(request: BrainV2CreateRequest | None = None):
    """IntegratedBrainV2 セッション作成。use_adex=true でAdExモデル。"""
    from src.brian2_circuits.integrated_brain_v2 import IntegratedBrainV2
    config = None
    if request and request.use_adex:
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        config = SharedCoreConfig(use_adex=True)
    brain = IntegratedBrainV2(config=config)
    from uuid import uuid4
    brain_id = f"v2-{uuid4().hex[:8]}"
    _brains_v2[brain_id] = brain
    model = "AdEx" if (request and request.use_adex) else "Izhikevich"
    return {"brain_id": brain_id, "model": model, "neurons": 821, "populations": 53}


@app.post("/brain/v2/{brain_id}/process")
def brain_v2_process(brain_id: str, input_data: BrainV2ProcessRequest):
    """IntegratedBrainV2 でテキスト処理。10情動状態を返す。"""
    if brain_id not in _brains_v2:
        raise HTTPException(404, f"Brain '{brain_id}' not found")
    result = _brains_v2[brain_id].process(input_data.text, context=input_data.context)
    return {
        "brain_id": brain_id,
        "step": result.step,
        "blocked": result.blocked,
        "emotion_state": result.emotion_state,
        "readout": {
            "valence": result.readout.valence,
            "arousal": result.readout.arousal,
            "threat_load": result.readout.threat_load,
        },
        "neuromodulation": result.neuromodulation,
        "theta_coherence": result.theta_coherence,
        "spiking_neurons": result.spiking_neurons,
    }


@app.post("/brain/v2/{brain_id}/sleep")
def brain_v2_sleep(brain_id: str, cycles: int = 1):
    """IntegratedBrainV2 の睡眠リプレイ。"""
    if brain_id not in _brains_v2:
        raise HTTPException(404, f"Brain '{brain_id}' not found")
    results = _brains_v2[brain_id].sleep(n_cycles=cycles)
    return {"brain_id": brain_id, "sleep_results": results}


@app.delete("/brain/v2/{brain_id}")
def delete_brain_v2(brain_id: str):
    """IntegratedBrainV2 セッション削除。"""
    if brain_id not in _brains_v2:
        raise HTTPException(404, f"Brain '{brain_id}' not found")
    del _brains_v2[brain_id]
    return {"status": "deleted"}


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.4.0", "models": ["izhikevich", "adex"]}
