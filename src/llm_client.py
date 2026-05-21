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

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import anthropic
from dotenv import load_dotenv


# ── 初始化：加载 .env ──────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

_DEFAULT_MODEL = "claude-sonnet-4-6"


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
        if self._backend == "anthropic":
            return self._anthropic_chat(messages, max_tokens, system)
        return self._openai_chat(messages, max_tokens, system)

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
        # 推理模型（DeepSeek-R1、MiniMax-M2.5 等）会将思考链包在 <think>...</think> 中。
        # 先处理正常闭合的情况（<think>...</think>answer），再处理未闭合（max_tokens 截断）
        if "<think>" in text:
            if "</think>" in text:
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            else:
                # think 块未闭合（token 耗尽），整段是推理链，无有效答案
                text = ""
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
