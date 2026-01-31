# Agent 工程师面试题库

## 目录
1. [基础概念题](#1-基础概念题)
2. [代码实现题](#2-代码实现题)
3. [系统设计题](#3-系统设计题)
4. [项目经验题](#4-项目经验题)
5. [开放性讨论题](#5-开放性讨论题)

---

## 1. 基础概念题

### Q1.1: 什么是 AI Agent？
**参考答案：**
```python
# AI Agent = 感知 + 决策 + 行动

class Agent:
    def __init__(self):
        self.perception = PerceptionModule()   # 感知环境
        self.brain = LLM()                    # 决策（LLM）
        self.action = ToolModule()            # 行动（工具）
        self.memory = MemoryModule()          # 记忆

    def run(self, task):
        # 循环：感知 → 思考 → 行动
        while not task.is_done():
            # 1. 感知当前状态
            state = self.perception.observe(task)

            # 2. 思考下一步行动
            thought = self.brain.reason(state, self.memory)

            # 3. 执行行动
            result = self.action.execute(thought)

            # 4. 更新记忆
            self.memory.update(thought, result)

        return task.result

# 关键特征：
# 1. 自主性：能独立做出决策
# 2. 反应性：能感知环境并做出反应
# 3. 主动性：能主动采取行动达成目标
# 4. 社交性：能与其他 Agent 或人类协作
```

---

### Q1.2: LLM Application 和 LLM Agent 的区别？

| 特性 | LLM Application | LLM Agent |
|------|-----------------|-----------|
| **决策方式** | 硬编码流程 | LLM 动态决策 |
| **能力边界** | LLM 本身能力 | 可通过工具扩展 |
| **与环境交互** | 无交互 | 能读取/修改环境 |
| **记忆** | 无或简单 | 复杂记忆系统 |
| **多步骤推理** | 有限 | 能规划和执行 |
| **示例** | Chatbot | AutoGPT, BabyAGI |

**代码对比：**
```python
# LLM Application（固定流程）
def chatbot(user_input):
    prompt = f"User: {user_input}\nAssistant:"
    response = llm.generate(prompt)
    return response

# LLM Agent（动态决策）
def agent(user_input):
    # Agent 自己决定需要做什么
    decision = llm.decide(user_input, available_tools)

    if decision.needs_tool:
        result = tool.execute(decision.tool_name, decision.tool_input)
        response = llm.generate(user_input, tool_result=result)
    else:
        response = llm.generate(user_input)

    return response
```

---

### Q1.3: 什么是 ReAct 框架？

**ReAct = Reasoning + Acting**

```python
def react_loop(agent, query, max_iterations=5):
    """
    ReAct 循环实现

    Reasoning: 思考下一步该做什么
    Acting: 执行相应的行动
    """

    # 初始状态
    thought = f"问题：{query}"

    for i in range(max_iterations):
        print(f"\n=== 迭代 {i+1} ===")
        print(f"思考: {thought}")

        # 1. Reasoning: LLM 决定下一步
        decision = agent.decide(thought)

        # 检查是否完成
        if decision.action == "finish":
            return decision.answer

        print(f"行动: {decision.action}({decision.input})")

        # 2. Acting: 执行行动
        if decision.action == "search":
            result = search_tool(decision.input)
        elif decision.action == "calculate":
            result = calculator_tool(decision.input)
        # ... 其他工具

        # 3. 观察：更新思考
        thought = f"上一步行动 {decision.action} 的结果是：{result}\n接下来应该？"

    return "无法在最大迭代次数内完成任务"

# ReAct 的优势：
# 1. 透明度高：可以看到推理过程
# 2. 可纠错：可以基于中间结果调整
# 3. 可扩展：容易添加新工具
```

---

### Q1.4: 什么是 Few-shot Prompting？

**答案：**
```python
# Zero-shot（零样本）：没有示例
prompt = "什么是AI？"

# One-shot（单样本）：一个示例
prompt = """
Q: 什么是机器学习？
A: 机器学习是让计算机从数据中学习的方法。

Q: 什么是AI？
A:
"""

# Few-shot（少样本）：多个示例
prompt = """
Q: 什么是机器学习？
A: 机器学习是让计算机从数据中学习的方法。

Q: 什么是深度学习？
A: 深度学习是使用神经网络的机器学习方法。

Q: 什么是强化学习？
A: 强化学习是通过奖励机制训练智能体的方法。

Q: 什么是AI？
A:
"""

# 在 Agent 中使用 Few-shot 提升工具调用准确率
def build_fewshot_prompt():
    return f"""你是一个助手，可以使用工具。

示例1:
用户: 读取 test.txt
助手: {{"tool": "read_file", "input": "test.txt"}}

示例2:
用户: 计算 1+1
助手: 直接回答，无需工具

示例3:
用户: 今天天气
助手: 直接回答，无需工具

当前用户: {user_input}
助手:"""
```

---

### Q1.5: LangChain 中的 Chain 和 Agent 区别？

**Chain（链）：**
```python
# Chain 是预定义的处理流程，每个环节的输入输出都是确定的
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

# 创建 Chain：Prompt → LLM → OutputParser
prompt = PromptTemplate(
    input_variables=["question"],
    template="回答这个问题：{question}"
)

llm = ChatOpenAI(model="gpt-4")

chain = LLMChain(llm=llm, prompt=prompt)

# 执行 Chain（确定性流程）
result = chain.run("什么是AI？")
# 流程：Prompt → LLM → 输出
```

**Agent（智能体）：**
```python
# Agent 根据输入动态决定下一步做什么
from langchain.agents import create_tool_calling_agent
from langchain.tools import Tool

tools = [
    Tool(name="search", func=search_func, description="搜索信息"),
    Tool(name="calculate", func=calc_func, description="数学计算")
]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 执行 Agent（动态决策）
result = agent.invoke({"input": "搜索 AI 的最新进展"})
# Agent 可能：search("AI progress") → LLM 总结 → 输出

result = agent.invoke({"input": "计算 123*456"})
# Agent 可能：calculate("123*456") → 输出
```

**总结：**
- **Chain**：适合流程固定的任务（如文本生成、摘要）
- **Agent**：适合需要灵活决策的任务（如多步骤推理）

---

## 2. 代码实现题

### Q2.1: 实现一个简单的 Tool 装饰器

**题目：** 实现一个类似 LangChain `@tool` 的装饰器

```python
# 解答
from typing import Callable, Dict, Any
import inspect

class Tool:
    """工具类"""

    def __init__(self, func: Callable, description: str = ""):
        self.func = func
        self.description = description or func.__doc__
        self.name = func.__name__

        # 提取函数签名
        sig = inspect.signature(func)
        self.args_schema = {
            name: {
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "any",
                "default": param.default if param.default != inspect.Parameter.empty else None
            }
            for name, param in sig.parameters.items()
        }

    def run(self, **kwargs):
        """执行工具"""
        return self.func(**kwargs)

def tool(description: str = ""):
    """
    工具装饰器

    使用示例：
    @tool(description="搜索信息")
    def search(query: str) -> str:
        return f"搜索结果：{query}"
    """
    def decorator(func):
        return Tool(func, description)

    return decorator

# 测试
@tool(description="两个数相加")
def add(a: int, b: int) -> int:
    """计算 a + b"""
    return a + b

# 使用
add_tool = add
print(add_tool.name)           # 'add'
print(add_tool.description)    # '两个数相加'
print(add_tool.args_schema)    # {'a': {...}, 'b': {...}}
print(add_tool.run(a=1, b=2))  # 3
```

---

### Q2.2: 实现 Agent 对话历史管理

**题目：** 实现一个支持滑动窗口的对话历史管理器

```python
# 解答
from collections import deque
from typing import List, Dict

class ConversationHistory:
    """对话历史管理器"""

    def __init__(self, max_turns: int = 10, max_tokens: int = 4000):
        """
        参数：
            max_turns: 最大保留轮数
            max_tokens: 最大 Token 数
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.history = deque(maxlen=max_turns)

    def add_message(self, role: str, content: str):
        """添加消息"""
        self.history.append({
            "role": role,
            "content": content,
            "tokens": len(content) // 2  # 估算 Token 数
        })

    def get_history(self) -> List[Dict]:
        """获取历史（按 Token 限制裁剪）"""
        # 从最近的开始累加 Token
        result = []
        total_tokens = 0

        for msg in reversed(self.history):
            if total_tokens + msg["tokens"] > self.max_tokens:
                break
            result.insert(0, msg)
            total_tokens += msg["tokens"]

        return result

    def get_history_string(self) -> str:
        """获取格式化的历史字符串"""
        history = self.get_history()
        lines = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        """清空历史"""
        self.history.clear()

# 测试
history = ConversationHistory(max_turns=5, max_tokens=100)

history.add_message("user", "你好")
history.add_message("assistant", "你好！有什么可以帮你的？")
history.add_message("user", "今天天气怎么样？")

print(history.get_history_string())
# 输出：
# 用户: 你好
# 助手: 你好！有什么可以帮你的？
# 用户: 今天天气怎么样？
```

---

### Q2.3: 实现 ReAct 循环

**题目：** 实现一个完整的 ReAct 循环

```python
# 解答
import json
import re
from typing import Dict, Any, Optional

class ReActAgent:
    """ReAct 框架的 Agent 实现"""

    def __init__(self, llm, tools: Dict[str, Callable]):
        self.llm = llm
        self.tools = tools
        self.max_iterations = 10

    def run(self, query: str) -> str:
        """执行 ReAct 循环"""

        # 初始 Prompt
        prompt = f"""回答以下问题：{query}

你可以使用以下工具：{list(self.tools.keys())}

请按以下格式回复：
思考：[你的思考过程]
行动：[工具名称] 输入=[工具输入]

或者如果不需要工具：
思考：[你的思考过程]
行动：finish 答案=[最终答案]
"""

        for iteration in range(self.max_iterations):
            print(f"\n=== 迭代 {iteration + 1} ===")

            # 1. LLM 生成回复
            response = self.llm.generate(prompt)
            print(f"LLM: {response}")

            # 2. 解析思考和行动
            thought = self._extract_thought(response)
            action = self._extract_action(response)

            print(f"思考: {thought}")
            print(f"行动: {action['type']}")

            # 3. 判断行动类型
            if action["type"] == "finish":
                return action["answer"]

            elif action["type"] in self.tools:
                # 4. 执行工具
                tool_func = self.tools[action["type"]]
                result = tool_func(action["input"])
                print(f"结果: {result}")

                # 5. 更新 Prompt
                prompt += f"\n{response}\n观察: {result}\n请继续："

            else:
                return f"未知行动：{action['type']}"

        return "超过最大迭代次数"

    def _extract_thought(self, response: str) -> str:
        """提取思考部分"""
        match = re.search(r"思考：(.+?)(?=\n|$)", response)
        return match.group(1).strip() if match else ""

    def _extract_action(self, response: str) -> Dict[str, Any]:
        """提取行动部分"""
        # 尝试匹配 "finish" 行动
        if "finish" in response.lower():
            match = re.search(r"答案[：=](.+)", response)
            return {"type": "finish", "answer": match.group(1).strip() if match else ""}

        # 尝试匹配工具调用
        match = re.search(r"行动[：:](\w+)\s+输入[=](.+)", response)
        if match:
            return {
                "type": match.group(1),
                "input": match.group(2).strip()
            }

        return {"type": "unknown"}

# 使用示例
def search_tool(query: str) -> str:
    return f"关于 '{query}' 的搜索结果"

llm = MockLLM()  # 假设的 LLM
tools = {"search": search_tool}

agent = ReActAgent(llm, tools)
result = agent.run("巴黎的人口是多少？")
```

---

### Q2.4: 实现工具缓存机制

**题目：** 实现 LRU 缓存的工具调用优化

```python
# 解答
from functools import lru_cache
import hashlib
import json
from typing import Any, Callable

class ToolCache:
    """工具结果缓存"""

    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._cache = {}
        self._access_order = []

    def _make_key(self, tool_name: str, tool_input: Any) -> str:
        """生成缓存键"""
        # 将输入序列化为字符串，然后哈希
        input_str = json.dumps(tool_input, sort_keys=True)
        return f"{tool_name}:{hashlib.md5(input_str.encode()).hexdigest()}"

    def get(self, tool_name: str, tool_input: Any) -> Optional[Any]:
        """获取缓存"""
        key = self._make_key(tool_name, tool_input)

        if key in self._cache:
            # 更新访问顺序（LRU）
            self._access_order.remove(key)
            self._access_order.append(key)

            print(f"[缓存命中] {tool_name}({tool_input})")
            return self._cache[key]

        return None

    def set(self, tool_name: str, tool_input: Any, result: Any):
        """设置缓存"""
        key = self._make_key(tool_name, tool_input)

        # 如果缓存已满，删除最久未使用的
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = result
        self._access_order.append(key)

        print(f"[缓存保存] {tool_name}({tool_input})")

    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._access_order.clear()

# 使用装饰器方式
def cached_tool(cache: ToolCache):
    """工具缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        def wrapper(tool_name: str, tool_input: Any):
            # 尝试从缓存获取
            cached = cache.get(tool_name, tool_input)
            if cached is not None:
                return cached

            # 缓存未命中，执行工具
            result = func(tool_input)
            cache.set(tool_name, tool_input, result)
            return result

        return wrapper
    return decorator

# 测试
cache = ToolCache(max_size=2)

@cached_tool(cache)
def expensive_computation(input_data: str) -> str:
    """模拟耗时计算"""
    print(f"[执行计算] {input_data}")
    return f"结果: {input_data}"

# 第一次调用 - 执行计算
expensive_computation("test1")
# 输出: [执行计算] test1
#       [缓存保存] test1

# 第二次调用 - 从缓存读取
expensive_computation("test1")
# 输出: [缓存命中] test1
```

---

## 3. 系统设计题

### Q3.1: 设计一个文档问答 Agent

**需求：**
- 用户上传 PDF 文档
- Agent 回答关于文档的问题
- 支持多文档问答
- 返回引用来源

**设计：**
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

class DocumentQAAgent:
    """文档问答 Agent"""

    def __init__(self):
        # 1. 文档加载器
        self.loader = PyPDFLoader()

        # 2. 文本分割器
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # 3. 向量数据库
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

        # 4. QA Chain
        self.qa_chain = None

    def upload_document(self, file_path: str):
        """上传文档"""
        # 加载 PDF
        documents = self.loader.load(file_path)

        # 分割文本
        texts = self.splitter.split_documents(documents)

        # 创建向量索引
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
        else:
            self.vector_store.add_documents(texts)

        # 创建 QA Chain
        from langchain.chains import RetrievalQA
        from langchain.chat_models import ChatOpenAI

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4"),
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def ask(self, question: str) -> Dict:
        """提问"""
        if self.qa_chain is None:
            return {"error": "请先上传文档"}

        # 执行问答
        result = self.qa_chain({"query": question})

        # 提取引用来源
        sources = [
            doc.metadata.get("source", "unknown")
            for doc in result.get("source_documents", [])
        ]

        return {
            "answer": result["result"],
            "sources": list(set(sources))
        }

# 使用
agent = DocumentQAAgent()
agent.upload_document("document.pdf")

response = agent.ask("文档主要讲了什么？")
print(response)
# {'answer': '文档主要讨论了...', 'sources': ['document.pdf']}
```

---

### Q3.2: 设计一个 Multi-Agent 研究助手

**需求：**
- 研究员 Agent：搜索和整理资料
- 写作者 Agent：撰写报告
- 审核者 Agent：检查质量
- 协调者 Agent：分配任务

**设计：**
```python
from typing import List
import asyncio

class Agent:
    """基础 Agent 类"""
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def execute(self, task: str) -> str:
        raise NotImplementedError

class ResearcherAgent(Agent):
    """研究员 Agent"""
    def execute(self, topic: str) -> str:
        print(f"[{self.name}] 正在研究：{topic}")
        # 搜索相关资料
        search_results = search_tool.search(topic)
        # 整理信息
        summary = f"关于 {topic} 的研究资料：{search_results}"
        return summary

class WriterAgent(Agent):
    """写作者 Agent"""
    def execute(self, material: str) -> str:
        print(f"[{self.name}] 正在撰写报告")
        # 基于材料生成报告
        report = llm.generate(f"基于以下材料写报告：{material}")
        return report

class CriticAgent(Agent):
    """审核者 Agent"""
    def execute(self, draft: str) -> str:
        print(f"[{self.name}] 正在审核报告")
        # 审核报告
        feedback = llm.generate(f"审核以下报告并提出改进意见：{draft}")
        return feedback

class CoordinatorAgent(Agent):
    """协调者 Agent"""
    def __init__(self):
        super().__init__("协调者", "协调和分配任务")
        self.researcher = ResearcherAgent("研究员1")
        self.writer = WriterAgent("写作者1")
        self.critic = CriticAgent("审核者1")

    def execute(self, task: str) -> str:
        print(f"[{self.name}] 开始任务：{task}")

        # 1. 研究阶段
        research_result = self.researcher.execute(task)

        # 2. 写作阶段
        draft = self.writer.execute(research_result)

        # 3. 审核和改进（循环直到满意）
        max_rounds = 3
        for i in range(max_rounds):
            feedback = self.critic.execute(draft)

            if "满意" in feedback or i == max_rounds - 1:
                break

            # 根据反馈修改
            draft = self.writer.execute(f"原稿：{draft}\n反馈：{feedback}")

        return draft

# 使用
coordinator = CoordinatorAgent()
final_report = coordinator.execute("人工智能的未来")
```

---

## 4. 项目经验题

### Q4.1: 描述一个你做过的 Agent 项目

**STAR 方法回答：**

**Situation（背景）：**
> "在之前的公司，我们需要一个自动化客服 Agent，能够回答用户关于产品的常见问题，并协助处理订单。"

**Task（任务）：**
> "我的任务是设计和实现一个基于 LLM 的客服 Agent，需要具备：
> - 理解用户问题
> - 查询知识库
> - 调用订单系统 API
> - 多轮对话能力
> - 学习用户反馈"

**Action（行动）：**
> "我采用了以下技术方案：
> 1. 使用 LangChain 框架构建 Agent
> 2. RAG 架构：向量数据库存储产品知识
> 3. 工具集成：订单查询、退款、库存查询
> 4. 记忆系统：保存用户上下文
> 5. 监控和日志：追踪 Agent 行为
>
> 遇到的挑战：
> - 幻觉问题：通过 RAG 和来源引用解决
> - API 延迟：实现异步工具调用
> - 成本优化：使用 GPT-3.5 进行初步筛选，GPT-4 处理复杂问题"

**Result（结果）：**
> "项目上线后：
> - 自动处理 70% 的客服咨询
> - 平均响应时间从 5 分钟降至 10 秒
> - 客户满意度提升 20%
> - 每月节省成本 10 万元
> - 团队规模减少 50%"

---

### Q4.2: 如何处理 Agent 的错误？

**答案：**
```python
class RobustAgent:
    """健壮的 Agent 实现"""

    def invoke(self, user_input: str) -> str:
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # 1. 执行 Agent
                response = self._execute_agent(user_input)

                # 2. 验证结果
                if self._validate_response(response):
                    return response

                # 3. 如果验证失败，请求 LLM 重新生成
                user_input = f"之前的回答 '{response}' 不正确，请重新回答：{user_input}"

            except json.JSONDecodeError as e:
                # JSON 解析失败
                print(f"[错误] JSON 解析失败，尝试重试 ({attempt + 1}/{max_retries})")
                # 让 LLM 重新生成，强调 JSON 格式
                user_input = f"请严格按照 JSON 格式回复：{user_input}"

            except Exception as e:
                print(f"[错误] 执行失败：{e}")
                # 降级策略：使用默认回答
                if attempt == max_retries - 1:
                    return self._get_fallback_response(user_input)

        return "抱歉，我遇到了一些问题，请稍后再试。"

    def _validate_response(self, response: str) -> bool:
        """验证响应是否合理"""
        # 检查长度
        if len(response) > 10000:
            return False

        # 检查敏感词
        forbidden_words = ["错误", "失败", "不能"]
        if any(word in response for word in forbidden_words):
            return False

        return True

    def _get_fallback_response(self, user_input: str) -> str:
        """降级回答"""
        return f"我暂时无法回答 '{user_input}'，请联系人工客服。"
```

---

## 5. 开放性讨论题

### Q5.1: 你认为 Agent 的未来发展方向是什么？

**参考要点：**
1. **多模态 Agent**：能处理文本、图像、音频、视频
2. **自主学习**：从经验中学习，不断改进
3. **协作 Agent**：多个 Agent 专门化协作
4. **具身智能**：Agent 拥有物理身体，能操作真实世界
5. **可解释性**：Agent 能解释自己的决策过程
6. **安全性**：确保 Agent 的行为可控、可预测

```python
# 未来 Agent 示例
class FutureAgent:
    def __init__(self):
        self.brain = LLM()                    # 大脑
        self.vision = VisionModel()          # 视觉
        self.speech = SpeechModel()          # 听说
        self.body = RobotBody()              # 身体
        self.memory = VectorMemory()         # 记忆
        self.learning_module = OnlineLearning()  # 学习

    def perceive(self):
        """多模态感知"""
        visual = self.vision.see()
        audio = self.speech.listen()
        return visual + audio

    def act(self):
        """具身行动"""
        return self.body.move()
```

---

### Q5.2: 如何评估 Agent 的性能？

**答案：**
```python
class AgentEvaluator:
    """Agent 性能评估器"""

    def evaluate(self, agent, test_cases: List[Dict]) -> Dict:
        """
        评估指标：
        1. 准确率：回答正确的比例
        2. 工具调用准确率：正确调用工具的比例
        3. 平均响应时间
        4. Token 消耗
        5. 成功率：完成任务的比例
        """
        results = {
            "correct": 0,
            "total": len(test_cases),
            "tool_accuracy": 0,
            "avg_time": 0,
            "total_tokens": 0,
            "success": 0
        }

        for case in test_cases:
            start = time.time()

            # 执行 Agent
            response = agent.invoke(case["input"])

            # 记录时间
            elapsed = time.time() - start
            results["avg_time"] += elapsed

            # 评估准确性
            if self._is_correct(response, case["expected_output"]):
                results["correct"] += 1

            # 评估工具调用
            if "tool_calls" in case:
                if self._check_tool_calls(response, case["tool_calls"]):
                    results["tool_accuracy"] += 1

            # 评估成功率
            if case["success_criteria"](response):
                results["success"] += 1

        # 计算平均值
        results["accuracy"] = results["correct"] / results["total"]
        results["tool_accuracy"] /= results["total"]
        results["avg_time"] /= results["total"]
        results["success_rate"] = results["success"] / results["total"]

        return results

# 测试用例示例
test_cases = [
    {
        "input": "读取 test.txt 并总结",
        "expected_output": "包含文件摘要",
        "tool_calls": [{"name": "read_file", "input": "test.txt"}],
        "success_criteria": lambda r: "总结" in r and len(r) > 50
    },
    # ... 更多测试用例
]
```

---

## 总结

### 面试准备清单

**理论基础：**
- [ ] 理解 Agent 基本概念
- [ ] 熟悉 ReAct 框架
- [ ] 掌握 Prompt Engineering
- [ ] 了解 RAG 架构

**编码能力：**
- [ ] 能够实现简单的 Agent
- [ ] 熟悉 LangChain API
- [ ] 理解异步编程
- [ ] 掌握缓存策略

**项目经验：**
- [ ] 至少 1-2 个完整项目
- [ ] 能讲清楚技术选型
- [ ] 了解常见问题和解决方案

**软技能：**
- [ ] 清晰的表达能力
- [ ] 逻辑思维
- [ ] 持续学习能力

**祝你面试顺利！** 🎯
