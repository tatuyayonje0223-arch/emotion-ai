"""FastAPI エンドポイントのテスト（セッションスコープ）。

[Codex adversarial-review fix: high]
旧テスト: グローバルシングルトンパイプラインのテスト。
修正: セッション作成→セッションID指定→セッション分離のテスト。
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, _manager


@pytest.fixture(autouse=True)
def reset_manager():
    """各テスト前にセッションマネージャをリセット。"""
    _manager._sessions.clear()
    yield
    _manager._sessions.clear()


client = TestClient(app)


def _create_session() -> str:
    """セッションを作成しIDを返す。"""
    resp = client.post("/sessions")
    assert resp.status_code == 200
    return resp.json()["session_id"]


class TestSessionManagement:
    def test_create_session(self):
        resp = client.post("/sessions")
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    def test_list_sessions(self):
        _create_session()
        _create_session()
        resp = client.get("/sessions")
        assert len(resp.json()["sessions"]) == 2

    def test_delete_session(self):
        sid = _create_session()
        resp = client.delete(f"/sessions/{sid}")
        assert resp.status_code == 200
        resp = client.get(f"/sessions/{sid}/state")
        assert resp.status_code == 404

    def test_nonexistent_session_404(self):
        resp = client.get("/sessions/nonexistent/state")
        assert resp.status_code == 404


class TestSessionScopedAPI:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_process_text(self):
        sid = _create_session()
        resp = client.post(f"/sessions/{sid}/process", json={"text": "嬉しいです！"})
        assert resp.status_code == 200
        data = resp.json()
        assert "state" in data
        assert "safety" in data

    def test_get_state(self):
        sid = _create_session()
        resp = client.get(f"/sessions/{sid}/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
        assert "valence" in data["state"]

    def test_tick(self):
        sid = _create_session()
        resp = client.post(f"/sessions/{sid}/tick")
        assert resp.status_code == 200

    def test_reset(self):
        sid = _create_session()
        client.post(f"/sessions/{sid}/process", json={"text": "test"})
        resp = client.post(f"/sessions/{sid}/reset")
        assert resp.status_code == 200
        assert resp.json()["step_count"] == 0

    def test_memory_query(self):
        sid = _create_session()
        client.post(f"/sessions/{sid}/process", json={"text": "素晴らしい！最高！嬉しい！"})
        resp = client.post(f"/sessions/{sid}/memory/query", json={"query_text": "素晴らしい"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == sid

    def test_audit_recent(self):
        sid = _create_session()
        client.post(f"/sessions/{sid}/process", json={"text": "テスト"})
        resp = client.get(f"/sessions/{sid}/audit/recent?limit=5")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == sid


class TestSessionIsolation:
    """[Codex fix] セッション間の状態分離を検証する。"""

    def test_sessions_are_isolated(self):
        """異なるセッションの状態が互いに影響しないこと。"""
        s1 = _create_session()
        s2 = _create_session()

        # s1にポジティブ入力
        client.post(f"/sessions/{s1}/process", json={"text": "嬉しい！最高！"})
        # s2にネガティブ入力
        client.post(f"/sessions/{s2}/process", json={"text": "悲しい。辛い。"})

        state1 = client.get(f"/sessions/{s1}/state").json()["state"]
        state2 = client.get(f"/sessions/{s2}/state").json()["state"]

        # s1とs2で異なるvalenceになっているはず
        assert state1["valence"] != state2["valence"]

    def test_reset_one_session_does_not_affect_other(self):
        s1 = _create_session()
        s2 = _create_session()
        client.post(f"/sessions/{s1}/process", json={"text": "テスト"})
        client.post(f"/sessions/{s2}/process", json={"text": "テスト"})

        # s1だけリセット
        client.post(f"/sessions/{s1}/reset")
        s1_state = client.get(f"/sessions/{s1}/state").json()
        s2_state = client.get(f"/sessions/{s2}/state").json()

        assert s1_state["step_count"] == 0
        assert s2_state["step_count"] > 0  # s2は影響されない

    def test_memory_isolated(self):
        s1 = _create_session()
        s2 = _create_session()
        client.post(f"/sessions/{s1}/process", json={"text": "秘密の情報です"})

        # s2から検索しても見つからない
        resp = client.post(f"/sessions/{s2}/memory/query", json={"query_text": "秘密"})
        results = resp.json()["results"]
        assert len(results) == 0
