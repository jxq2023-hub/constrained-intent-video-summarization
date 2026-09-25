"""
意图理解模块 - Intent Understanding Module
核心创新点：从自然语言提取结构化意图

基于intend_demo.py优化，集成到Agent系统
"""

import json
import re
from typing import Dict, List
from dataclasses import dataclass, asdict


@dataclass
class Intent:
    """结构化意图对象"""
    goal: str  # quick_browse, semantic_extract, scene_analysis, adaptive
    focus: List[str]  # 关注的对象/场景/动作
    constraints: Dict[str, any]  # 约束条件
    output_format: str  # video, images, description
    confidence: float = 1.0
    reasoning: str = ""  # LLM的推理过程

    def to_dict(self):
        return asdict(self)


class IntentParser:
    """意图解析器 - 使用LLM的CoT策略"""

    def __init__(
            self,
            api_key: str = "ollama",
            model: str = "qwen2.5:7b",
            base_url: str = "http://127.0.0.1:11434/v1",
            temperature: float = 0.1,
            debug: bool = False,
            focus_vocabulary: List[str] = None,
    ):
        self.debug = debug
        self.focus_vocabulary = list(focus_vocabulary or [])

        # 初始化LLM (OpenAI兼容接口，支持Ollama)
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            self._model = model
            self._temperature = temperature
            if debug:
                print(f"意图解析器初始化成功 (模型: {model}, 接口: {base_url})")
        except Exception as e:
            raise RuntimeError(f"LLM初始化失败: {e}")

        # Few-shot示例库
        self.examples = [
            {
                "query": "找出猫跳跃的精彩瞬间，做成30秒短视频",
                "reasoning": "用户想要提取特定内容（猫跳跃），有明确的关注对象和动作，且指定了时长约束（30秒）和输出格式（短视频）",
                "intent": {
                    "goal": "semantic_extract",
                    "focus": ["猫", "跳跃"],
                    "constraints": {"duration": 30, "prefer_dynamic": True},
                    "output_format": "video"
                }
            },
            {
                "query": "快速浏览这个1小时的会议录像",
                "reasoning": "用户想要快速了解视频内容，没有特定关注点，目标是概览整个视频",
                "intent": {
                    "goal": "quick_browse",
                    "focus": [],
                    "constraints": {"brief": True},
                    "output_format": "video"
                }
            },
            {
                "query": "提取所有人物特写镜头，大概15帧",
                "reasoning": "用户想要提取特定类型的镜头（人物特写），指定了帧数要求（15帧）",
                "intent": {
                    "goal": "semantic_extract",
                    "focus": ["人物", "特写"],
                    "constraints": {"frame_count": 15},
                    "output_format": "images"
                }
            }
        ]

    def parse(self, query: str) -> Intent:
        """解析用户意图（使用CoT策略）"""
        if self.debug:
            print(f"\n解析用户输入: {query}")

        try:
            # 构建CoT prompt
            prompt = self._build_cot_prompt(query)

            # 调用LLM (OpenAI兼容接口)
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=1000,
            )
            raw_output = response.choices[0].message.content.strip()

            if self.debug:
                print(f"LLM输出:\n{raw_output[:200]}...")

            # 解析输出
            intent = self._parse_output(raw_output)
            intent = self._postprocess_intent(query, intent)

            if self.debug:
                print(f"意图: {intent.goal}, 关注: {intent.focus}, 置信度: {intent.confidence:.2f}")

            return intent

        except Exception as e:
            if self.debug:
                print(f"解析失败: {e}")
            # 返回默认意图
            return Intent(
                goal='adaptive',
                focus=[],
                constraints={},
                output_format='video',
                confidence=0.3
            )

    def _postprocess_intent(self, query: str, intent: Intent) -> Intent:
        """Apply deterministic fixes for common under-specified LLM outputs."""
        # When a controlled downstream ontology is active, an explicit focus
        # means the request is query-specific even if the sentence contrasts it
        # with a "general overview".  This enforces the same decision boundary
        # stated in the prompt; it does not alter or repair the extracted labels.
        if self.focus_vocabulary and intent.focus and intent.goal in {'adaptive', 'quick_browse'}:
            intent.goal = 'semantic_extract'

        if intent.goal == 'semantic_extract' and not intent.focus:
            if any(keyword in query for keyword in ['精彩', '高光']):
                intent.focus = ['精彩片段']
            elif any(keyword in query for keyword in ['关键', '重点']):
                intent.focus = ['关键片段']
            elif any(keyword in query for keyword in ['找出', '提取', '查找']):
                intent.focus = ['目标内容']

            if intent.focus:
                intent.confidence = self._calculate_confidence(
                    intent.goal,
                    intent.focus,
                    intent.constraints
                )

        if '关键帧视频' in query or '关键帧短片' in query:
            intent.output_format = 'video'
        elif any(keyword in query for keyword in ['关键帧', '帧图片', '预览图片', '保存图片', '导出图片']):
            intent.output_format = 'images'
        elif any(keyword in query for keyword in ['视频', '短片', '片段', '剪辑']):
            intent.output_format = 'video'

        return intent

    def _build_cot_prompt(self, query: str) -> str:
        """构建Chain-of-Thought prompt"""
        prompt = """你是视频摘要意图理解专家。请仔细分析用户需求，先思考再输出结构化意图。

## 分析步骤

1. **理解用户目标**: 用户想要什么类型的摘要？
   - quick_browse: 快速浏览、概览
   - semantic_extract: 提取特定内容
   - scene_analysis: 场景分析
   - adaptive: 通用摘要

   **判定边界**: 只要用户明确指定了关注的对象、人物、场景或动作，
   即使句子中使用了“摘要”“精华”或“简短视频”等词，也应判定为
   semantic_extract；只有没有任何特定关注点的通用摘要请求才使用 adaptive。

2. **识别关注点**: 用户关注哪些对象、场景或动作？

3. **提取约束**: 用户有什么限制？
   - duration: 时长限制（秒）
   - frame_count: 帧数要求
   - diversity: 需要多样性
   - prefer_dynamic: 偏好动态内容

4. **确定输出格式**: video（视频）、images（图片）、description（描述）

## 示例分析

"""
        # 添加few-shot示例
        for i, ex in enumerate(self.examples[:2], 1):
            prompt += f"""示例{i}:
用户输入: "{ex['query']}"
思考过程: {ex['reasoning']}
结构化意图:
{json.dumps(ex['intent'], ensure_ascii=False, indent=2)}

"""

        prompt += f"""## 现在请分析

用户输入: "{query}"

请按以下格式输出：

思考过程: [你的分析]

结构化意图:
[JSON格式的意图对象]

注意：JSON必须严格按照格式，所有字段都必须包含。"""

        if self.focus_vocabulary:
            vocabulary = "、".join(self.focus_vocabulary)
            prompt += f"""

## 受控关注概念词表

后续视频模型只接受以下规范概念：{vocabulary}

如果用户使用同义词、描述性短语或口语表达，请先理解其含义，再把 focus
规范化为词表中语义最接近的概念。不要把原始描述直接复制到 focus；无法可靠
映射的内容不要强行添加。"""

        return prompt

    def _parse_output(self, raw_output: str) -> Intent:
        """解析LLM输出"""
        # 提取reasoning
        reasoning = ""
        reasoning_match = re.search(r'思考过程[:：]\s*(.+?)(?=结构化意图|$)',
                                    raw_output, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # 提取JSON
        json_str = self._extract_json(raw_output)

        if not json_str:
            raise ValueError("无法从输出中提取JSON")

        # 解析JSON
        intent_dict = json.loads(json_str)

        # 验证和标准化
        return self._validate_intent(intent_dict, reasoning)

    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON对象"""
        # 移除markdown代码块
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # 找到最外层的大括号
        stack = []
        start_idx = -1

        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx != -1:
                        json_str = text[start_idx:i + 1]
                        try:
                            json.loads(json_str)
                            return json_str
                        except:
                            continue

        raise ValueError("无法提取有效的JSON")

    def _validate_intent(self, data: Dict, reasoning: str = "") -> Intent:
        """验证并创建Intent对象"""
        # 验证goal
        valid_goals = ['quick_browse', 'semantic_extract', 'scene_analysis', 'adaptive']
        goal = data.get('goal', 'adaptive')
        if goal not in valid_goals:
            goal = 'adaptive'

        # 验证focus
        focus = data.get('focus', [])
        if isinstance(focus, str):
            focus = [focus] if focus else []
        elif not isinstance(focus, list):
            focus = []
        focus = [f.strip() for f in focus if isinstance(f, str) and f.strip()]

        # 验证constraints
        constraints = data.get('constraints', {})
        if not isinstance(constraints, dict):
            constraints = {}

        # 类型转换
        if 'duration' in constraints:
            try:
                constraints['duration'] = int(constraints['duration'])
            except (ValueError, TypeError):
                del constraints['duration']

        if 'frame_count' in constraints:
            try:
                constraints['frame_count'] = int(constraints['frame_count'])
            except (ValueError, TypeError):
                del constraints['frame_count']

        # 验证output_format
        valid_formats = ['video', 'images', 'description']
        output_format = data.get('output_format', 'video')
        if output_format not in valid_formats:
            output_format = 'video'

        # 计算置信度
        confidence = self._calculate_confidence(goal, focus, constraints)

        return Intent(
            goal=goal,
            focus=focus,
            constraints=constraints,
            output_format=output_format,
            confidence=confidence,
            reasoning=reasoning
        )

    def _calculate_confidence(self, goal: str, focus: List[str],
                              constraints: Dict) -> float:
        """计算意图解析的置信度"""
        confidence = 1.0

        # 如果是semantic_extract但没有focus，降低置信度
        if goal == 'semantic_extract' and not focus:
            confidence *= 0.7

        # 如果constraints为空，略微降低置信度
        if not constraints:
            confidence *= 0.95

        return confidence
