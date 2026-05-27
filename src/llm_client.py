#!/usr/bin/env python3
"""
LLM 配置与客户端统一入口

所有模块通过 get_client() 获取 LLMClient 实例，通过统一的
.chat(messages, max_tokens, system=None) 接口调用 LLM，
无需关心底层是 Anthropic 还是 OpenAI 兼容接口。

使用方法：
  from src.llm_client import get_client, get_model, LLMClient

  client: LLMClient = get_client()
  text = client.chat(
      messages=[{"role": "user", "content": "..."}],
      max_tokens=512,
  )

配置来源（.env 文件，参考 .env.example）：

  Anthropic 模式（默认，不设置 LLM_BASE_URL）：
    ANTHROPIC_API_KEY=sk-ant-xxx
    LLM_MODEL=claude-sonnet-4-6

  OpenAI 兼容模式（GLM / MiniMax / GPT / DeepSeek 等）：
    LLM_API_KEY=your-api-key
    LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
    LLM_MODEL=glm-4

后端自动检测规则：
  LLM_BASE_URL 已设置 → OpenAI 兼容后端（需要 openai 包）
  LLM_BASE_URL 未设置 → Anthropic 原生后端（需要 anthropic 包）
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import anthropic
from dotenv import load_dotenv


# ── 初始化：加载 .env ──────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

_DEFAULT_MODEL = "claude-sonnet-4-6"


# ── 异常与重试配置 ────────────────────────────────────────────────────────────

class LLMTruncationError(RuntimeError):
    """
    推理模型思维链（<think>）未闭合即耗尽 max_tokens，或返回空内容，无有效输出。

    这是**可恢复错误**：上层（chat_json）可通过增大 max_tokens 重试。
    禁止静默吞为空字符串（违反"禁止兜底"原则）。
    """


class LLMJSONError(ValueError):
    """
    LLM 输出无法解析为 JSON（已穷尽有界重试）。

    继承 ValueError 以兼容既有 `except ValueError` 调用方。
    """


# 瞬时网络/服务错误的重试配置（退避秒数 = base ** attempt）
_TRANSIENT_MAX_RETRIES = 3
_TRANSIENT_BACKOFF_BASE = 2.0


def _is_transient_error(exc: Exception) -> bool:
    """
    判断异常是否为可重试的瞬时错误（网络断开/超时/限流/服务端 5xx）。

    按异常类名匹配，避免硬依赖某一 SDK 的异常类型层级。
    """
    name = type(exc).__name__
    transient_names = {
        "APIConnectionError", "APITimeoutError", "RateLimitError",
        "InternalServerError", "ServiceUnavailableError", "RemoteProtocolError",
    }
    return name in transient_names or isinstance(exc, (ConnectionError, TimeoutError))


def _extract_json_block(raw: str) -> Dict[str, Any]:
    """
    从 LLM 原始输出中提取并解析 JSON 对象。

    依次处理：```json fence → ``` fence → 裸 {...}（MiniMax 等无 fence 风格）。

    参数：
        raw: LLM 原始输出文本

    返回：
        解析后的 dict

    异常：
        ValueError:            输入为空
        json.JSONDecodeError:  无法解析为合法 JSON
    """
    if not raw or not raw.strip():
        raise ValueError("LLM 输出为空，无 JSON 可解析")
    s = raw.strip()
    if "```json" in s:
        start = s.index("```json") + 7
        end = s.rindex("```")
        s = s[start:end].strip()
    elif "```" in s:
        start = s.index("```") + 3
        end = s.rindex("```")
        s = s[start:end].strip()
    if not s.startswith("{"):
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            s = match.group(0)
    return json.loads(s)


# ── 统一 LLM 客户端 ───────────────────────────────────────────────────────────

class LLMClient:
    """
    统一 LLM 客户端，屏蔽 Anthropic 和 OpenAI 兼容接口的差异。

    不直接构造此类，通过 get_client() 工厂函数获取实例。

    核心方法：
      .chat(messages, max_tokens, system=None) → str
        messages: [{"role": "user"/"assistant", "content": "..."}]
        system:   系统提示（Anthropic: 顶层参数；OpenAI-compat: 前置 system 消息）

    属性：
      .model: 当前使用的模型名称（从 LLM_MODEL 读取）
    """

    def __init__(
        self,
        model: str,
        backend: Literal["anthropic", "openai"],
        anthropic_client: Optional[anthropic.Anthropic] = None,
        openai_client: Optional[Any] = None,
    ) -> None:
        """
        参数：
            model:            模型 ID 字符串
            backend:          "anthropic" 或 "openai"
            anthropic_client: Anthropic 客户端实例（backend="anthropic" 时使用）
            openai_client:    OpenAI 客户端实例（backend="openai" 时使用）
        """
        self.model = model
        self._backend = backend
        self._anthropic = anthropic_client
        self._openai = openai_client

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        system: Optional[str] = None,
    ) -> str:
        """
        统一 LLM 调用接口。

        参数：
            messages:   对话消息列表，格式：
                        [{"role": "user"/"assistant", "content": "..."}]
            max_tokens: 最大生成 token 数
            system:     系统提示文本（可选）
                        Anthropic 后端：作为顶层 system 参数传入
                        OpenAI 兼容后端：转为 {"role":"system",...} 前置于 messages

        返回：
            模型输出文本（已调用 .strip()）

        异常：
            任何 API 异常直接传播（不静默捕获）
        """
        last_exc: Optional[Exception] = None
        for attempt in range(_TRANSIENT_MAX_RETRIES):
            try:
                if self._backend == "anthropic":
                    return self._anthropic_chat(messages, max_tokens, system)
                return self._openai_chat(messages, max_tokens, system)
            except Exception as exc:
                # 仅对瞬时错误重试；LLMTruncationError 等非瞬时错误立即上抛
                if _is_transient_error(exc) and attempt < _TRANSIENT_MAX_RETRIES - 1:
                    wait = _TRANSIENT_BACKOFF_BASE ** attempt
                    print(
                        f"[LLMClient] 瞬时错误 {type(exc).__name__}，"
                        f"{wait:.0f}s 后重试 ({attempt + 1}/{_TRANSIENT_MAX_RETRIES})",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    last_exc = exc
                    continue
                raise
        # 理论不可达：循环要么 return 要么 raise
        raise RuntimeError(f"[LLMClient] 重试逻辑异常退出: {last_exc}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        system: Optional[str] = None,
        max_retries: int = 2,
    ) -> Tuple[Dict[str, Any], str]:
        """
        调用 LLM 并将输出解析为 JSON dict，带有界重试与修复。

        恢复策略（非兜底：全程可观测、最终失败仍抛错）：
          - 截断（LLMTruncationError）→ 增大 max_tokens 重试
          - JSON 解析失败 → 追加"仅输出 JSON、抑制推理"指令后重试
          - 穷尽 max_retries 次仍失败 → 抛 LLMJSONError（含原始输出供审计）

        参数：
            messages:    对话消息（最后一条须为 user 角色）
            max_tokens:  初始 max_tokens（重试时按 1.5x 递增，上限 64000）
            system:      系统提示
            max_retries: 额外重试次数（总尝试次数 = max_retries + 1）

        返回：
            (data, raw): 解析后的 dict 与原始输出文本

        异常：
            LLMJSONError: 穷尽重试仍无法解析为合法 JSON
        """
        cur_max_tokens = max_tokens
        last_raw = ""
        last_err: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            msgs = messages
            if attempt > 0:
                # 重试：抑制思维链长度，强制仅输出 JSON
                repair = (
                    "\n\nIMPORTANT: Output ONLY a single valid JSON object. "
                    "Do NOT include any reasoning, explanation, or markdown fences."
                )
                msgs = list(messages[:-1]) + [
                    {**messages[-1], "content": messages[-1]["content"] + repair}
                ]
                cur_max_tokens = min(int(cur_max_tokens * 1.5), 64000)

            try:
                raw = self.chat(messages=msgs, max_tokens=cur_max_tokens, system=system)
            except LLMTruncationError as exc:
                last_err, last_raw = exc, ""
                print(
                    f"[LLMClient.chat_json] 截断，重试 {attempt + 1}/{max_retries}"
                    f"（max_tokens→{cur_max_tokens}）",
                    file=sys.stderr,
                )
                continue

            last_raw = raw
            try:
                return _extract_json_block(raw), raw
            except (json.JSONDecodeError, ValueError) as exc:
                last_err = exc
                print(
                    f"[LLMClient.chat_json] JSON 解析失败，重试 {attempt + 1}/{max_retries}",
                    file=sys.stderr,
                )
                continue

        raise LLMJSONError(
            f"[LLMClient.chat_json] 穷尽 {max_retries} 次重试仍无法解析 JSON。\n"
            f"  最后原因: {type(last_err).__name__}: {last_err}\n"
            f"  原始输出（前 400 字）: {last_raw[:400]!r}"
        )

    def _anthropic_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        system: Optional[str],
    ) -> str:
        """
        Anthropic 原生接口调用。

        system 参数作为顶层字段传入（Anthropic 的标准用法）。
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        response = self._anthropic.messages.create(**kwargs)
        return response.content[0].text.strip()

    def _openai_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        system: Optional[str],
    ) -> str:
        """
        OpenAI 兼容接口调用（适用于 GLM / MiniMax / GPT / DeepSeek 等）。

        system 提示转为 {"role": "system", "content": ...} 前置于 messages。
        """
        all_messages: List[Dict[str, str]] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = self._openai.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=all_messages,
        )
        text = response.choices[0].message.content
        if text is None:
            raise LLMTruncationError(
                "[LLMClient] OpenAI 兼容接口返回 content=None（无有效输出）"
            )
        # 推理模型（DeepSeek-R1、MiniMax-M2.5 等）会将思考链包在 <think>...</think> 中。
        # 先处理正常闭合的情况（<think>...</think>answer），再处理未闭合（max_tokens 截断）
        if "<think>" in text:
            if "</think>" in text:
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            else:
                # think 块未闭合（max_tokens 耗尽）：不再静默吞为空串（违反禁止兜底），
                # 显式抛出可恢复的截断错误，由 chat_json/上层增大 max_tokens 后重试。
                raise LLMTruncationError(
                    "[LLMClient] 推理模型 <think> 未闭合即耗尽 max_tokens，无有效输出。"
                    "请增大 max_tokens 后重试。"
                )
        if not text.strip():
            raise LLMTruncationError(
                "[LLMClient] OpenAI 兼容接口返回空文本（疑似截断或无输出）"
            )
        return text


# ── 工厂函数 ──────────────────────────────────────────────────────────────────

def get_model() -> str:
    """
    获取系统统一使用的 LLM 模型名称。

    从 LLM_MODEL 环境变量读取，未设置时使用默认值。

    返回：
        模型 ID 字符串（如 "claude-sonnet-4-6"）
    """
    return os.environ.get("LLM_MODEL") or _DEFAULT_MODEL


def get_embedding_model() -> str:
    """
    获取 Dense 向量检索的 embedding 模型名称。

    返回：
        模型名（SentenceTransformers 格式）
    """
    return os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_client() -> LLMClient:
    """
    构建并返回 LLMClient 实例。

    后端自动检测：
      LLM_BASE_URL 已设置 → OpenAI 兼容后端
        必填：LLM_API_KEY、LLM_BASE_URL、LLM_MODEL
        依赖：pip install openai
      LLM_BASE_URL 未设置 → Anthropic 原生后端
        必填：ANTHROPIC_API_KEY
        可选：LLM_MODEL（默认 claude-sonnet-4-6）

    返回：
        LLMClient 实例

    异常：
        ValueError:   必填环境变量缺失
        ImportError:  使用 OpenAI 兼容模式但未安装 openai 包
    """
    model = get_model()
    base_url = os.environ.get("LLM_BASE_URL")

    if base_url:
        # OpenAI 兼容模式（GLM / MiniMax / GPT / DeepSeek 等）
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "[llm_client] 使用 OpenAI 兼容模式（LLM_BASE_URL 已设置）时，"
                "必须同时设置 LLM_API_KEY。\n"
                "请在 .env 文件中添加：LLM_API_KEY=your-api-key"
            )
        try:
            import openai as _openai
            import httpx as _httpx
        except ImportError:
            raise ImportError(
                "[llm_client] 使用 OpenAI 兼容模式需要安装 openai 包：\n"
                "  pip install openai"
            )
        # httpx 支持 socks5:// 但不支持无版本号的 socks:// scheme。
        # 若系统 ALL_PROXY 使用 socks:// 格式，将其归一化为 socks5://，
        # 避免 httpx 初始化时抛出 "Unknown scheme for proxy URL" 错误。
        proxy_url = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("ALL_PROXY")
            or os.environ.get("all_proxy")
        )
        if proxy_url and proxy_url.startswith("socks://"):
            proxy_url = "socks5://" + proxy_url[len("socks://"):]
        http_client = _httpx.Client(proxy=proxy_url) if proxy_url else None
        openai_client = _openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        return LLMClient(model=model, backend="openai", openai_client=openai_client)

    # Anthropic 原生模式（默认）
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "[llm_client] 未找到 ANTHROPIC_API_KEY。\n"
            "请在 .env 文件中设置（参考 .env.example）：\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
            "若使用其他 LLM 提供商，请设置 LLM_BASE_URL + LLM_API_KEY。"
        )
    anthropic_client = anthropic.Anthropic(api_key=api_key)
    return LLMClient(model=model, backend="anthropic", anthropic_client=anthropic_client)


def print_config() -> None:
    """
    打印当前 LLM 配置摘要（隐藏 key 的敏感部分）。

    用于实验开始时确认配置，方便复现。
    """
    base_url = os.environ.get("LLM_BASE_URL")

    if base_url:
        api_key = os.environ.get("LLM_API_KEY", "")
        key_display = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "(未设置)"
        backend_label = "OpenAI 兼容"
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        key_display = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "(未设置)"
        backend_label = "Anthropic 原生"
        base_url = "https://api.anthropic.com（默认）"

    print("── LLM 配置 ─────────────────────────────────")
    print(f"  后端:              {backend_label}")
    print(f"  API Key:           {key_display}")
    print(f"  Base URL:          {base_url}")
    print(f"  LLM_MODEL:         {get_model()}")
    print(f"  EMBEDDING_MODEL:   {get_embedding_model()}")
    print("──────────────────────────────────────────────")
