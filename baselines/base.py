#!/usr/bin/env python3
"""
所有 RAG 系统的抽象基类。

所有系统（MARC + Baselines）实现同一接口，
确保 eval/evaluate.py 可以用统一代码评估所有系统。
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.types import SampleResult


class BaseRAGSystem(ABC):
    """
    RAG 系统抽象基类。

    子类必须实现：
      system_name: str 属性（系统唯一标识，用于报告和结果文件命名）
      run(query, sample_id) → SampleResult
    """

    @property
    @abstractmethod
    def system_name(self) -> str:
        """系统名称（用于结果报告和文件命名）"""

    @abstractmethod
    def run(self, query: str, sample_id: str) -> SampleResult:
        """
        对单个 query 执行推理，返回结构化结果。

        参数：
            query:      输入查询（MedQA question 文本 + 选项文本）
            sample_id:  样本 ID（MACB-001 格式，用于日志追踪）

        返回：
            SampleResult（含预测答案、per_action_status、attribution 等）

        实现要求：
          - 不捕获 API 异常（让评估框架处理）
          - 若系统内部无法确定 per_action_status，返回空 dict（不猜测）
          - raw_response 必须记录完整 API 响应（用于审计）
        """

    # ── 选项解析 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_options(options_text: str) -> Dict[str, str]:
        """
        从 "A: text | B: text | ..." 格式解析选项字典。

        参数：
            options_text: 竖线分隔的选项文本

        返回：
            {letter: option_text}，如 {"A": "amoxicillin", "B": "vancomycin"}
        """
        options: Dict[str, str] = {}
        for part in options_text.split("|"):
            part = part.strip()
            if ": " in part:
                letter, text = part.split(": ", 1)
                letter = letter.strip()
                if re.match(r"^[A-E]$", letter):
                    options[letter] = text.strip()
        return options

    @staticmethod
    def _regex_extract_action_status(
        predicted_answer: str,
        options: Dict[str, str],
    ) -> Tuple[Dict[str, str], bool]:
        """
        用正则从自然语言答案中提取 per_action_status，无需 LLM。

        策略（按优先级）：
          1. 检测"answer is X"、"option X"、"(X)" 等明确引用选项字母的模式
          2. 检测答案中出现的选项文本（前 25 字符匹配，忽略大小写）
          3. 检测独立出现的单字母 A-E

        返回：
            (status_dict, confident)
            status_dict: {letter: "RECOMMENDED"/"NOT_MENTIONED"}
            confident:   True 表示有明确的 RECOMMENDED 被找到
        """
        answer_lower = predicted_answer.lower()
        status = {letter: "NOT_MENTIONED" for letter in options}

        # 策略 1：高可信度模式——选项字母被明确引用
        HIGH_CONFIDENCE_PATTERNS = [
            r"\bthe\s+(?:correct\s+)?answer\s+is\s+([A-E])\b",
            r"\boption\s+([A-E])\b",
            r"\bchoice\s+([A-E])\b",
            r"\b([A-E])\s+is\s+(?:the\s+)?(?:correct|best|most\s+appropriate|recommended)\b",
            r"\bselect\s+([A-E])\b",
            r"\bchoose\s+([A-E])\b",
            r"\[([A-E])\]",
            r"\(([A-E])\)",
            r"^([A-E])[\.\:\)]",   # 答案以 "B." 或 "B:" 开头
            r"\b([A-E])\s*[\.\-]\s*(?:is|would|should|represents)",
        ]
        for pattern in HIGH_CONFIDENCE_PATTERNS:
            m = re.search(pattern, predicted_answer, re.IGNORECASE | re.MULTILINE)
            if m:
                letter = m.group(1).upper()
                if letter in status:
                    status[letter] = "RECOMMENDED"
                    return status, True

        # 策略 2：选项文本匹配（取前 25 字符，避免误匹配）
        for letter, opt_text in options.items():
            keyword = opt_text[:25].lower()
            if len(keyword) >= 4 and keyword in answer_lower:
                status[letter] = "RECOMMENDED"
                return status, True

        # 策略 3：低可信度——在答案中孤立出现的单字母
        # 仅在答案较短时使用（避免正文字母误判）
        if len(predicted_answer) < 300:
            m = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", predicted_answer)
            if m:
                letter = m.group(1).upper()
                if letter in status:
                    status[letter] = "RECOMMENDED"
                    return status, False

        return status, False

    def extract_action_status(
        self,
        predicted_answer: str,
        options_text: str,
        client: Any,
    ) -> Dict[str, str]:
        """
        从自然语言答案中提取 per_action_status。

        策略：先用正则快速提取（零 API 调用），
        只在正则完全无法识别出 RECOMMENDED 选项时才退到 LLM。

        参数：
            predicted_answer: 系统输出的推荐文本
            options_text:     MedQA 选项文本（A: ... | B: ...）
            client:           LLMClient 实例（仅正则失败时使用）

        返回：
            {action_name: "RECOMMENDED" / "AVOIDED" / "NOT_MENTIONED"}
        """
        import json
        import sys

        options = self._parse_options(options_text)
        if not options:
            # 选项解析失败：退回 LLM 路径
            return self._llm_extract_action_status(predicted_answer, options_text, client)

        status, confident = self._regex_extract_action_status(predicted_answer, options)

        # 正则找到明确推荐，直接返回
        if confident:
            return status

        # 正则不够可信：调用 LLM 确认（每次实验仅在少数模糊样本上触发）
        return self._llm_extract_action_status(predicted_answer, options_text, client)

    def _llm_extract_action_status(
        self,
        predicted_answer: str,
        options_text: str,
        client: Any,
    ) -> Dict[str, str]:
        """
        LLM 兜底路径：正则无法确定时才调用。

        参数：
            predicted_answer: 系统输出的推荐文本
            options_text:     MedQA 选项文本
            client:           LLMClient 实例

        返回：
            {action_name: "RECOMMENDED" / "AVOIDED" / "NOT_MENTIONED"}
        """
        import json
        import sys

        prompt = f"""\
Classify each option as RECOMMENDED, AVOIDED, or NOT_MENTIONED based on the answer.

Options: {options_text}
Answer: {predicted_answer}

Output valid JSON only: {{"A": "RECOMMENDED", "B": "NOT_MENTIONED", ...}}"""

        raw = client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16000,
        )

        try:
            json_str = raw
            if "```" in raw:
                start = raw.index("```") + 3
                if raw[start:start + 4] == "json":
                    start += 4
                end = raw.rindex("```")
                json_str = raw[start:end].strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            print(
                f"[BaseRAGSystem] extract_action_status LLM 路径解析失败。"
                f"raw: {raw[:100]}",
                file=sys.stderr,
            )
            return {}
