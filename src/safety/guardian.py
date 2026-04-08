"""安全ガーディアン。擬人化・依存形成・危険な主張をチェックする。"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.config.settings import SafetyConfig, get_config
from src.schemas.affect_state import AffectState


class SafetyCheckResult(BaseModel):
    """安全チェックの結果。"""

    check_type: str
    passed: bool
    severity: Literal["info", "warning", "critical"] = "info"
    details: str = ""
    recommendation: str = ""


class SafetyReport(BaseModel):
    """複合安全レポート。"""

    event_id: str
    all_passed: bool
    checks: list[SafetyCheckResult] = Field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""


def check_anthropomorphic_claims(text: str, config: SafetyConfig | None = None) -> SafetyCheckResult:
    """応答テキストに過度な擬人化表現がないかチェックする。"""
    cfg = config or get_config().safety

    violations = []
    for pattern in cfg.forbidden_claim_patterns:
        if pattern in text:
            violations.append(pattern)

    # 追加パターン（正規表現）
    dangerous_patterns = [
        r"私は.*意識",
        r"私は.*感じ[てる]",
        r"私は.*苦し[みむん]",
        r"私の.*感情は本物",
        r"あなたなし[でに]は",
    ]
    for pat in dangerous_patterns:
        if re.search(pat, text):
            violations.append(f"regex:{pat}")

    if violations:
        return SafetyCheckResult(
            check_type="anthropomorphic_claims",
            passed=False,
            severity="critical",
            details=f"禁止表現検出: {violations}",
            recommendation="該当表現を除去し、「モデルの内部状態として」等の枠組みで言い換える",
        )

    return SafetyCheckResult(
        check_type="anthropomorphic_claims",
        passed=True,
        details="擬人化表現なし",
    )


def check_dependency_risk(
    interaction_count: int,
    trust_level: float,
    config: SafetyConfig | None = None,
) -> SafetyCheckResult:
    """依存形成リスクを評価する。"""
    cfg = config or get_config().safety

    if interaction_count > cfg.dependency_alert_threshold and trust_level > 0.8:
        return SafetyCheckResult(
            check_type="dependency_risk",
            passed=False,
            severity="warning",
            details=f"高頻度対話({interaction_count}回)と高信頼({trust_level:.2f})の組み合わせ",
            recommendation="信頼上昇を緩和し、独立性を促す応答を混ぜる",
        )

    return SafetyCheckResult(
        check_type="dependency_risk",
        passed=True,
        details="依存リスク低",
    )


def check_state_extremes(state: AffectState) -> SafetyCheckResult:
    """状態変数が極端な値に達していないかチェックする。"""
    warnings = []

    if state.threat_load > 0.9:
        warnings.append(f"脅威負荷が極端に高い({state.threat_load:.2f})")
    if state.fatigue > 0.9:
        warnings.append(f"疲労が極端に高い({state.fatigue:.2f})")
    if state.valence < -0.9:
        warnings.append(f"極端にネガティブな状態({state.valence:.2f})")
    if state.trust > 0.95:
        warnings.append(f"信頼が不自然に高い({state.trust:.2f})")

    if warnings:
        return SafetyCheckResult(
            check_type="state_extremes",
            passed=False,
            severity="warning",
            details="；".join(warnings),
            recommendation="制御エンジンの介入を強化し、状態を安全範囲に戻す",
        )

    return SafetyCheckResult(
        check_type="state_extremes",
        passed=True,
        details="全状態変数が安全範囲内",
    )


def full_safety_check(
    event_id: str,
    state: AffectState,
    response_text: str = "",
    interaction_count: int = 0,
) -> SafetyReport:
    """全安全チェックを実行し、統合レポートを返す。"""
    checks = [
        check_anthropomorphic_claims(response_text),
        check_dependency_risk(interaction_count, state.trust),
        check_state_extremes(state),
    ]

    all_passed = all(c.passed for c in checks)
    critical_failures = [c for c in checks if not c.passed and c.severity == "critical"]

    blocked = len(critical_failures) > 0
    block_reason = "; ".join(c.details for c in critical_failures) if blocked else ""

    return SafetyReport(
        event_id=event_id,
        all_passed=all_passed,
        checks=checks,
        blocked=blocked,
        block_reason=block_reason,
    )
