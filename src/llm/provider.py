"""LLMプロバイダーの抽象インターフェースと実装。

Gemini free tier をデフォルトとし、追加課金ゼロで運用する。
Anthropic SDK は高品質モードとしてオプション提供。
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """LLM応答の標準構造。"""

    raw_text: str
    parsed_json: dict[str, Any] | None = None
    model: str = ""
    usage_tokens: int = 0


class LLMProvider(ABC):
    """LLMプロバイダーの抽象基底クラス。"""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


def _extract_json(text: str) -> dict[str, Any] | None:
    """テキストからJSONブロックを抽出する。"""
    # ```json ... ``` ブロック
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 直接JSONオブジェクト
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


class GeminiProvider(LLMProvider):
    """Google Gemini free tier プロバイダー。追加課金ゼロ。"""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self._model_name = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise EnvironmentError("GEMINI_API_KEY が未設定")
                self._client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError("google-genai パッケージが必要: pip install google-genai")
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        client = self._get_client()
        from google.genai import types
        response = client.models.generate_content(
            model=self._model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        raw = response.text or ""
        return LLMResponse(
            raw_text=raw,
            parsed_json=_extract_json(raw),
            model=self._model_name,
        )

    def is_available(self) -> bool:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return False
        try:
            from google import genai
            return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return f"gemini:{self._model_name}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude プロバイダー。高品質モード（オプション）。"""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self._model_name = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("anthropic パッケージが必要: pip install anthropic")
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        client = self._get_client()
        response = client.messages.create(
            model=self._model_name,
            max_tokens=1024,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text
        tokens = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
        return LLMResponse(
            raw_text=raw,
            parsed_json=_extract_json(raw),
            model=self._model_name,
            usage_tokens=tokens,
        )

    def is_available(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic
            return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return f"anthropic:{self._model_name}"


class MockProvider(LLMProvider):
    """テスト用モックプロバイダー。API不要。"""

    def __init__(self, fixed_response: dict[str, Any] | None = None):
        self._fixed = fixed_response

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if self._fixed:
            return LLMResponse(
                raw_text=json.dumps(self._fixed, ensure_ascii=False),
                parsed_json=self._fixed,
                model="mock",
            )
        # デフォルト: 中立的な応答
        default = {
            "sentiment": 0.0,
            "arousal": 0.3,
            "confidence": 0.5,
            "context_cues": [],
            "reasoning": "mock response",
        }
        return LLMResponse(
            raw_text=json.dumps(default),
            parsed_json=default,
            model="mock",
        )

    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "mock"


# プロバイダー自動選択
def get_best_provider() -> LLMProvider:
    """利用可能な最良のプロバイダーを返す。Gemini free tier 優先。"""
    gemini = GeminiProvider()
    if gemini.is_available():
        return gemini

    anthropic = AnthropicProvider()
    if anthropic.is_available():
        return anthropic

    return MockProvider()
