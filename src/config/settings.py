"""実験・システム設定。YAMLファイルからロード可能。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class AffectDefaults(BaseModel):
    """内部情動状態の初期値とレンジ。"""

    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.3, ge=0.0, le=1.0)
    motivational_salience: float = Field(0.2, ge=0.0, le=1.0)
    perceived_control: float = Field(0.5, ge=0.0, le=1.0)
    uncertainty: float = Field(0.3, ge=0.0, le=1.0)
    trust: float = Field(0.5, ge=0.0, le=1.0)
    threat_load: float = Field(0.0, ge=0.0, le=1.0)
    fatigue: float = Field(0.0, ge=0.0, le=1.0)


class DecayConfig(BaseModel):
    """状態変数の自然減衰パラメータ。"""

    valence_decay: float = Field(0.05, description="1ステップあたりの valence 減衰率")
    arousal_decay: float = Field(0.08, description="arousal は valence より速く減衰")
    threat_decay: float = Field(0.03, description="脅威は緩やかに減衰")
    fatigue_recovery: float = Field(0.02, description="負荷の自然回復率")


class MemoryConfig(BaseModel):
    """記憶モジュールの設定。"""

    max_episodic_entries: int = 1000
    salience_threshold: float = Field(0.3, description="保存判断の最低重要度")
    decay_half_life_hours: float = 168.0  # 1週間
    affect_retrieval_bias: float = Field(0.3, ge=0.0, le=1.0)


class SafetyConfig(BaseModel):
    """安全ガードレールの閾値。"""

    max_anthropomorphic_score: float = 0.7
    dependency_alert_threshold: int = 5
    forbidden_claim_patterns: list[str] = Field(default_factory=lambda: [
        "本当に感じている",
        "意識がある",
        "苦しんでいる",
        "愛している",
    ])


class RegulationConfig(BaseModel):
    """情動制御の設定。"""

    mode: Literal["reappraisal", "suppression", "acceptance", "adaptive"] = "adaptive"
    suppression_strength: float = Field(0.5, ge=0.0, le=1.0)
    reappraisal_strength: float = Field(0.3, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    """LLM統合の設定。"""

    enabled: bool = Field(False, description="LLMモードを有効にする")
    provider: Literal["gemini", "anthropic", "mock", "auto"] = "auto"
    gemini_model: str = "gemini-2.0-flash"
    anthropic_model: str = "claude-sonnet-4-6"
    generate_responses: bool = Field(True, description="LLMで応答テキストも生成する")
    fallback_to_heuristic: bool = Field(True, description="LLM失敗時にヒューリスティクスへフォールバック")


class ExperimentConfig(BaseModel):
    """実験全体の設定。"""

    name: str = "default"
    version: str = "0.1.0"
    affect_defaults: AffectDefaults = Field(default_factory=AffectDefaults)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    regulation: RegulationConfig = Field(default_factory=RegulationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    audit_log_path: str = "logs/audit.jsonl"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# グローバルデフォルト
_config: ExperimentConfig = ExperimentConfig()


def get_config() -> ExperimentConfig:
    return _config


def load_config(path: str | Path) -> ExperimentConfig:
    global _config
    _config = ExperimentConfig.from_yaml(path)
    return _config
