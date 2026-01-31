# Agent 开发学习指南

## 目录
1. [项目整体架构](#1-项目整体架构)
2. [核心概念解析](#2-核心概念解析)
3. [代码深度解析](#3-代码深度解析)
4. [Agent 开发核心技能](#4-agent-开发核心技能)
5. [面试高频问题](#5-面试高频问题)

---

## 1. 项目整体架构

### 1.1 什么是 Tool-Driven Agent？

```
┌─────────────┐
│   用户输入   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│    LLM (大脑)           │
│  - 理解用户意图          │
│  - 决策是否需要工具      │
└──────┬──────────────────┘
       │
       ▼ (如果需要工具)
┌─────────────────────────┐
│   Tool 调用层           │
│  - read_file()          │
│  - 其他自定义工具        │
└──────┬──────────────────┘
       │
       ▼ (工具返回结果)
┌─────────────────────────┐
│    LLM 再次处理         │
│  - 基于工具结果          │
│  - 生成最终回答          │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│   输出给用户            │
└─────────────────────────┘
```

### 1.2 项目文件结构

```
agent_tool_demo/
├── agent.py           # Agent 核心逻辑（重点）
├── tools.py           # 工具定义
├── main.py            # 程序入口
├── test.txt           # 测试数据（论文）
├── requirements.txt   # 依赖管理
├── .env              # 环境变量配置
└── README.md         # 项目说明
```

---

## 2. 核心概念解析

### 2.1 Agent 的三大核心能力

```python
# 1. Planning (规划能力)
# 将复杂任务分解为子目标
"写一篇论文" → ["搜集资料", "写大纲", "写正文", "修改"]

# 2. Memory (记忆能力)
# 存储和检索历史信息
conversation_history = [
    {"role": "user", "content": "读取test.txt"},
    {"role": "assistant", "content": "已读取文件..."}
]

# 3. Tool Use (工具使用)
# 调用外部工具扩展能力
tools = {
    "read_file": {"func": read_file_func, "description": "..."}
}
```

### 2.2 ReAct 框架（本项目使用）

```
ReAct = Reasoning (推理) + Acting (行动)

循环过程：
1. Thought (思考): 我需要读取文件
2. Action (行动): 调用 read_file("test.txt")
3. Observation (观察): 获得文件内容
4. Thought (思考): 内容是关于智能制造的
5. Action (行动): 生成回答
```

### 2.3 LangChain 生态

```
langchain-core      # 核心抽象层
├── LLM             # 大语言模型接口
├── Tools           # 工具抽象
├── Agents          # Agent 实现
└── Memory          # 记忆管理

langchain           # 高级API
langchain-openai    # OpenAI 集成
langchain-community # 社区集成
```

---

## 3. 代码深度解析

### 3.1 tools.py - 工具定义

```python
from langchain.tools import tool

@tool  # LangChain 装饰器，自动将函数转换为工具
def read_file(path: str) -> str:
    """
    Read the content of a text file.

    Args:
        path: 文件路径

    Returns:
        文件内容字符串

    工具的三要素：
    1. 名称: read_file
    2. 描述: 自动从 docstring 提取
    3. 函数: 实际执行逻辑
    """
    if not os.path.exists(path):
        return "File not found"

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# @tool 装饰器做了什么？
# 1. 创建 StructuredTool 对象
# 2. 提取函数签名作为输入模式
# 3. 提取 docstring 作为工具描述
# 4. 包装函数为可调用对象
```

**关键点：**
- `@tool` 装饰器是 LangChain 的核心机制
- Docstring 非常重要！LLM 通过它理解工具用途
- 工具应该有清晰的类型注解和错误处理

---

### 3.2 agent.py - Agent 核心逻辑

#### 3.2.1 Agent 初始化

```python
class SimpleAgent:
    def __init__(self, llm, tools):
        self.llm = llm                           # 大脑
        self.tools = tools                       # 工具箱
        self.conversation_history = []           # 短期记忆
        self.tool_results = {}                   # 工具结果缓存

    # 设计模式：单例 + 状态管理
```

**设计要点：**
- **LLM 作为决策中心**：所有推理都由 LLM 完成
- **工具结果缓存**：避免重复调用相同工具
- **对话历史管理**：实现多轮对话的上下文

---

#### 3.2.2 invoke 方法 - 核心执行流程

```python
def invoke(self, inputs):
    user_input = inputs.get("input", "")

    # 步骤1：记录用户输入
    self.conversation_history.append({
        "role": "user",
        "content": user_input
    })

    # 步骤2：构建 Prompt（包含历史和上下文）
    system_prompt = self._build_prompt(user_input)

    # 步骤3：LLM 第一次推理（决定是否调用工具）
    response = self.llm.invoke([HumanMessage(content=system_prompt)])
    response_text = response.content

    # 步骤4：解析 LLM 响应，判断是否需要工具
    if "{" in response_text and "}" in response_text:
        tool_call = json.loads(response_text)
        tool_name = tool_call.get("tool")
        tool_input = tool_call.get("input")

        # 步骤5：调用工具
        if tool_name in self.tools:
            tool_result = self.tools[tool_name]["func"](tool_input)

            # 步骤6：将工具结果返回给 LLM
            follow_up_prompt = self._build_followup_prompt(
                tool_name, tool_input, tool_result, user_input
            )
            final_response = self.llm.invoke([HumanMessage(content=follow_up_prompt)])
            return {"output": final_response.content}

    # 步骤7：直接返回 LLM 回答（无需工具）
    return {"output": response_text}
```

**流程图：**
```
用户输入
   ↓
构建 Prompt (包含历史 + 工具描述)
   ↓
LLM 推理
   ↓
包含 JSON？ ← 否 → 直接返回回答
   ↓ 是
解析工具调用
   ↓
执行工具 → 获得结果
   ↓
构建新 Prompt (包含工具结果)
   ↓
LLM 生成最终回答
   ↓
返回给用户
```

---

#### 3.2.3 Prompt 工程技巧

```python
def _build_prompt(self, user_input):
    base_prompt = f"""你是一个有帮助的助手，可以使用工具来完成任务。

{self.tool_descriptions}  # ← 告诉 LLM 有哪些工具

当你需要使用工具时，请按以下 JSON 格式回复：
{{"tool": "工具名称", "input": "工具输入"}}

【重要】如果用户询问"文章"、"文件"、"内容"等，
且之前已经读取过 test.txt 文件，请基于对话历史中已保存的内容回答。
"""

    # 关键优化：将已读取的文件内容添加到上下文
    if "read_file(test.txt)" in self.tool_results:
        file_content = self.tool_results["read_file(test.txt)"]
        base_prompt += f"\n【已读取文件内容 - test.txt】\n{file_content}\n\n"

    # 添加对话历史
    if len(self.conversation_history) > 0:
        base_prompt += "\n【对话历史】\n"
        recent_history = self.conversation_history[-10:]
        for msg in recent_history:
            if msg["role"] in ["user", "assistant"]:
                role = "用户" if msg["role"] == "user" else "助手"
                base_prompt += f"{role}: {msg['content']}\n"

    return base_prompt
```

**Prompt 设计原则：**
1. **清晰的角色定义**：告诉 LLM 它是谁
2. **明确的工具说明**：工具描述要详细
3. **结构化的输出格式**：JSON 格式便于解析
4. **上下文注入**：相关信息要放在 Prompt 中
5. **历史管理**：保留最近 N 轮对话

---

### 3.3 main.py - 程序入口

```python
def build_agent():
    # 1. 初始化 LLM
    llm = ChatOpenAI(
        model="glm-4-flash",           # 智谱模型
        temperature=0,                 # 温度=0 更确定
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    # 2. 定义工具
    tools = {
        "read_file": {
            "func": read_file.func,      # ← 注意：.func 获取实际函数
            "description": "读取文本文件的内容。输入：文件路径（str）。输出：文件内容（str）。"
        }
    }

    # 3. 创建 Agent
    return SimpleAgent(llm, tools)
```

**关键配置：**
- `temperature=0`：输出更确定，适合工具调用
- `base_url`：支持兼容 OpenAI 的其他 API
- `read_file.func`：LangChain 工具需要提取底层函数

---

## 4. Agent 开发核心技能

### 4.1 技术栈要求

```
必备技能：
├── Python 编程 (熟练)
│   ├── 类型注解 (Type Hints)
│   ├── 装饰器 (Decorators)
│   └── 异步编程 (Async/Await)
│
├── LangChain 生态
│   ├── langchain-core (核心概念)
│   ├── langchain (高级API)
│   └── 集成各种 LLM (OpenAI, Claude, etc.)
│
├── Prompt Engineering
│   ├── Few-shot prompting
│   ├── Chain-of-Thought
│   └── ReAct 框架
│
└── AI/LLM 基础
    ├── Transformer 架构
    ├── Token 概念
    └── API 调用优化
```

### 4.2 常用 Agent 模式

```python
# 模式1: ReAct Agent (本项目使用)
Thought → Action → Observation → Thought → ...

# 模式2: Plan-and-Execute
Planning (分解任务) → Execute (逐步执行)

# 模式3: Multi-Agent
多个 Agent 协作：Researcher + Writer + Critic

# 模式4: Self-Reflection
Agent 自我反思和改进
```

### 4.3 工具开发最佳实践

```python
from langchain.tools import tool
from typing import Optional

# ✅ 好的工具设计
@tool
def search_database(
    query: str,
    limit: int = 10,
    filters: Optional[dict] = None
) -> str:
    """
    在数据库中搜索信息。

    Args:
        query: 搜索关键词，必填
        limit: 返回结果数量，默认10
        filters: 额外过滤条件，可选

    Returns:
        JSON 格式的搜索结果
    """
    # 实现逻辑
    pass

# ❌ 不好的工具设计
@tool
def do_something(data):  # 缺少类型注解
    """做一些事情"""       # 描述不清晰
    # 实现逻辑
    pass
```

**工具设计原则：**
1. **单一职责**：一个工具只做一件事
2. **清晰描述**：docstring 要详细说明用途和参数
3. **错误处理**：优雅处理异常情况
4. **类型注解**：帮助 LLM 理解参数类型
5. **返回格式化**：结果易于 LLM 理解

---

## 5. 面试高频问题

### Q1: 什么是 Agent？它和 Chatbot 有什么区别？

**答案：**
```python
# Chatbot (对话机器人)
用户输入 → LLM → 输出
(纯对话，无外部动作)

# Agent (智能体)
用户输入 → LLM 决策 → 工具调用 → LLM 总结 → 输出
(能执行动作，改变环境)

关键区别：
1. Agent 能使用工具
2. Agent 有记忆（短期+长期）
3. Agent 能规划多步骤任务
4. Agent 能与环境交互
```

### Q2: 如何处理 Agent 的幻觉问题？

**答案：**
```python
# 方法1: RAG (检索增强生成)
def agent_with_rag(query):
    # 先从知识库检索相关文档
    relevant_docs = vector_db.search(query, top_k=3)
    # 将文档作为上下文
    prompt = f"背景知识：{relevant_docs}\n问题：{query}"
    return llm.generate(prompt)

# 方法2: 工具验证
def verify_response(response):
    # 使用工具验证 LLM 的输出
    if needs_tool(response):
        result = tool.execute(response)
        return llm.refine(response, result)

# 方法3: 思维链 (CoT)
prompt = f"让我们一步步思考：\n{question}"
# 引导 LLM 展示推理过程
```

### Q3: 如何优化 Agent 的性能？

**答案：**
```python
# 1. 并行工具调用
async def parallel_tools():
    results = await asyncio.gather(
        tool1.execute(),
        tool2.execute(),
        tool3.execute()
    )
    return results

# 2. 缓存机制
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_tool(input_data):
    # 缓存工具结果，避免重复调用
    return slow_operation(input_data)

# 3. Prompt 优化
# 使用 Few-shot 示例
prompt = """
示例1:
用户: 读取test.txt
助手: {"tool": "read_file", "input": "test.txt"}

示例2:
用户: 今天天气
助手: 直接回答，无需工具

现在请处理: {user_input}
"""

# 4. Token 优化
# 只保留必要的上下文
def optimize_context(history, max_tokens=2000):
    while count_tokens(history) > max_tokens:
        history.pop(0)  # 移除最旧的对话
    return history
```

### Q4: LangChain 中的 Chain 和 Agent 有什么区别？

**答案：**
```python
# Chain (链)
# 预定义的步骤序列，每步的输入输出都是确定的
chain = (
    PromptTemplate() |
    LLM() |
    OutputParser()
)
# 适合：流程固定的任务

# Agent (智能体)
# 根据输入动态决定下一步做什么
agent = create_tool_calling_agent(
    llm=llm,
    tools=[tool1, tool2, tool3],
    prompt=prompt
)
# 适合：需要灵活决策的任务
```

### Q5: 如何设计一个多 Agent 系统？

**答案：**
```python
# 层级式 Multi-Agent
class MultiAgentSystem:
    def __init__(self):
        # 主控 Agent：规划和分配任务
        self.coordinator = Agent(
            role="协调者",
            goal="分解任务并分配给专业Agent"
        )

        # 专业 Agents
        self.researcher = Agent(role="研究员", tools=["search", "read"])
        self.writer = Agent(role="写作者", tools=["write"])
        self.critic = Agent(role="评论者", tools=["analyze"])

    def execute(self, task):
        # 步骤1: 主控分解任务
        subtasks = self.coordinator.plan(task)

        # 步骤2: 并行执行子任务
        research_result = self.researcher.run(subtasks[0])
        draft = self.writer.run(subtasks[1], research_result)

        # 步骤3: 评审和改进
        feedback = self.critic.review(draft)
        final = self.writer.revise(draft, feedback)

        return final
```

### Q6: 实际项目中遇到过什么挑战？如何解决的？

**参考答案：**
```python
# 挑战1: 工具调用解析失败
# 问题：LLM 返回的 JSON 格式不正确
# 解决：
def robust_parse(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # 使用正则表达式提取 JSON
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        # 失败时要求 LLM 重试
        return {"tool": None, "error": "Invalid format"}

# 挑战2: Token 超限
# 解决：滑动窗口 + 重要性评分
def smart_truncate(history, max_tokens):
    # 保留最近 N 轮对话
    recent = history[-5:]
    # 保留包含关键信息的对话
    important = [h for h in history if contains_keywords(h)]
    return recent + important

# 挑战3: 工具执行失败
# 解决：重试机制 + 降级策略
def execute_with_retry(tool, input_data, max_retries=3):
    for i in range(max_retries):
        try:
            return tool.execute(input_data)
        except Exception as e:
            if i == max_retries - 1:
                # 最后一次失败，返回友好提示
                return f"工具执行失败：{str(e)}"
            # 重试前稍微修改输入
            input_data = modify_input(input_data)
```

---

## 6. 进阶学习路线

### 6.1 入门阶段（1-2周）
- [ ] 理解 Agent 基本概念
- [ ] 熟悉 LangChain 基础API
- [ ] 完成本项目的学习
- [ ] 实现简单的工具调用Agent

### 6.2 进阶阶段（1个月）
- [ ] 学习 RAG（检索增强生成）
- [ ] 实现多工具Agent
- [ ] 掌握 Prompt Engineering
- [ ] 学习向量数据库（Pinecone, Milvus）

### 6.3 高级阶段（2-3个月）
- [ ] Multi-Agent 系统
- [ ] Agent 监控和调试
- [ ] 性能优化技巧
- [ ] 生产环境部署

### 6.4 推荐资源
```
官方文档：
- LangChain Docs: https://python.langchain.com/
- OpenAI Cookbook: https://cookbook.openai.com/

书籍：
- "Prompt Engineering Guide"
- "Building Applications with LLMs"

论文：
- ReAct: "Reasoning and Acting"
- Chain-of-Thought: "Prompting Methods"

实践项目：
1. 文档问答系统（RAG）
2. 代码助手Agent
3. 数据分析Agent
4. 跨Agent协作系统
```

---

## 7. 面试准备清单

### 7.1 技术准备
- [ ] 能够手写简单的 Agent 代码
- [ ] 理解 LangChain 的核心概念
- [ ] 熟悉常用的 Agent 模式
- [ ] 有实际的 Agent 项目经验

### 7.2 算法准备
- [ ] JSON 解析和格式化
- [ ] 字符串处理和正则表达式
- [ ] 异步编程基础
- [ ] 缓存策略

### 7.3 项目准备
准备 1-2 个完整的项目，能够讲清楚：
1. 项目的业务场景
2. 为什么选择 Agent 方案
3. 遇到的技术挑战
4. 如何优化和改进
5. 项目的数据流和架构图

### 7.4 常见代码题
```python
# 题目1: 实现一个简单的 ReAct loop
def react_loop(agent, initial_query, max_steps=5):
    query = initial_query
    for step in range(max_steps):
        response = agent.think(query)
        if response.action == "finish":
            return response.answer
        result = agent.execute(response.action, response.input)
        query = f"Previous action: {response.action}, Result: {result}"

# 题目2: 实现工具缓存
from functools import lru_cache

class ToolCache:
    def __init__(self):
        self.cache = {}

    def get(self, tool_name, tool_input):
        key = f"{tool_name}:{tool_input}"
        if key in self.cache:
            return self.cache[key]
        result = execute_tool(tool_name, tool_input)
        self.cache[key] = result
        return result
```

---

## 8. 总结

### 8.1 Agent 开发的核心要点

```python
# 1. LLM 是大脑，负责决策
# 2. 工具是手，负责执行
# 3. Prompt 是语言，与 LLM 沟通
# 4. 记忆是笔记本，保存上下文

class Agent:
    def __init__(self):
        self.brain = LLM()        # 思考
        self.hands = Tools()      # 行动
        self.memory = Memory()    # 记忆

    def run(self, task):
        # 思考 → 行动 → 观察 → 思考 ...
        while not task.done():
            thought = self.brain.think(task, self.memory)
            action = self.brain.decide_action(thought)
            result = self.hands.execute(action)
            self.memory.update(action, result)
            task.progress(result)
        return task.result
```

### 8.2 项目关键代码回顾

```python
# agent.py 的核心逻辑
1. 接收用户输入
2. 构建 Prompt（包含历史、工具描述、文件内容）
3. LLM 推理（决定是否调用工具）
4. 如果需要工具 → 解析 JSON → 调用工具
5. 将工具结果返回给 LLM
6. LLM 生成最终答案
7. 保存到对话历史
```

### 8.3 下一步学习建议

1. **动手实践**：修改本项目，添加更多工具
2. **阅读源码**：深入学习 LangChain 源码
3. **关注前沿**：关注 AI Agent 领域最新论文
4. **构建作品集**：完成 2-3 个有挑战性的项目

---

**祝你学习顺利，成为优秀的 Agent 工程师！** 🚀
