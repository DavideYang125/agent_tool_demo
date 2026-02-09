# Agent 开发入门：从零构建你的第一个智能体

## 前言

你是否想过 ChatGPT 是如何"思考"的？它不仅能回答问题，还能：
- 📄 读取文件内容
- 🔍 搜索网络信息
- 💻 执行代码
- 📊 查询数据库

这些能力的背后，都有一个核心概念：**Agent（智能体）**。

在这个系列中，我将带你从零开始，深入理解 Agent 的开发。今天我们先从基础开始：**什么是 Agent？如何从零实现一个简单的 Agent？**

**本文的特色**：我们不依赖 LangChain 的封装，而是**从零实现**一个 Agent，让你真正理解 Agent 的核心原理！

---

## 一、Agent 是什么？

### 1.1 传统 LLM 的局限

先看看传统的大语言模型（LLM）：

```python
# 传统 LLM 只能"说"，不能"做"
user_input = "读取 test.txt 文件的内容"

response = llm.generate(user_input)
# 输出：抱歉，我无法访问文件系统
```

**问题**：LLM 被限制在文本生成领域，无法执行实际操作。

### 1.2 Agent 的解决方案

Agent 的核心思想：**让 LLM 学会使用工具**。

```python
# Agent 可以"思考"并"行动"
user_input = "读取 test.txt 文件的内容"

# Agent 的思考过程
thought = "用户想读取文件，我需要调用 read_file 工具"
action = '{"tool": "read_file", "input": "test.txt"}'
observation = "# 欢迎使用 Agent\n\n这是一个示例文件..."

# 最终返回结果
response = "文件内容如下：\n# 欢迎使用 Agent..."
```

### 1.3 Agent vs 传统 LLM

| 特性 | 传统 LLM | Agent |
|------|----------|-------|
| 能力 | 纯文本生成 | 思考 + 行动 |
| 数据源 | 训练数据 | 工具/API/文件系统 |
| 自主性 | 被动回答 | 主动决策 |
| 记忆 | 无状态 | 可以保存上下文 |

---

## 二、Agent 的核心：ReAct 框架

### 2.1 什么是 ReAct？

**ReAct** = **Re**asoning（推理）+ **Act**ing（行动）

这是 Agent 最常用的思考模式，核心流程：

```
┌─────────────────────────────────────────────┐
│  1. Thought（思考）：用户想要什么？           │
│  2. Action（行动）：我需要调用哪个工具？      │
│  3. Observation（观察）：工具返回了什么？     │
│  4. Answer（回答）：基于结果回答用户          │
└─────────────────────────────────────────────┘
```

### 2.2 实际例子

用户问题：**"读取 test.txt 文件的内容"**

```
🤖 Thought: 用户想读取文件，我需要使用 read_file 工具
🔧 Action: {"tool": "read_file", "input": "test.txt"}
📄 Observation: 工具返回文件内容...
✅ Answer: 已成功读取文件，内容是...
```

---

## 三、项目结构

```
agent_tool_demo/
├── agent.py              # Agent 核心实现
├── tools.py              # 工具定义
├── main.py               # 程序入口
├── .env                  # API Key 配置
└── test.txt              # 测试文件
```

---

## 四、从零实现 Agent

### 4.1 第一步：定义工具

首先，我们需要给 Agent 一些"能力"：

```python
# tools.py

from langchain.tools import tool
import os

@tool
def read_file(path: str) -> str:
    """
    读取文本文件的内容

    Args:
        path: 文件路径

    Returns:
        文件的文本内容
    """
    # 检查文件是否存在
    if not os.path.exists(path):
        return "文件不存在"

    # 读取文件内容
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
```

**关键点**：
- 使用 `@tool` 装饰器，LangChain 会自动生成工具的 schema
- 工具必须有清晰的文档字符串，LLM 会根据它理解工具用途
- 工具返回字符串，便于 LLM 处理

**`@tool` 装饰器做了什么？**

```python
# @tool 装饰器会自动包装函数
print(read_file.func)        # 实际的函数
print(read_file.name)        # "read_file"
print(read_file.description) # 工具描述
print(read_file.args_schema) # 参数的 JSON Schema
```

### 4.2 第二步：实现 Agent 核心

这是整个项目的核心：**从零实现一个带记忆的 Agent**。

```python
# agent.py

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 加载环境变量
load_dotenv()

class SimpleAgent:
    """带记忆的 Tool-Driven Agent 实现"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.tool_descriptions = self._format_tool_descriptions()

        # 对话历史（核心！）
        self.conversation_history = []

        # 工具调用记录（保存工具返回的结果）
        self.tool_results = {}

    def _format_tool_descriptions(self):
        """格式化工具描述，让 LLM 理解可用工具"""
        desc = "\n可用工具：\n"
        for name, tool_info in self.tools.items():
            desc += f"- {name}: {tool_info['description']}\n"
        return desc

    def invoke(self, inputs):
        """处理用户输入（带对话记忆）"""
        user_input = inputs.get("input", "")

        # 1. 保存用户消息到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # 2. 构建系统提示（包含对话历史）
        system_prompt = self._build_prompt(user_input)

        # 3. 第一次调用 LLM
        response = self.llm.invoke([HumanMessage(content=system_prompt)])
        response_text = response.content

        # 4. 检查是否需要调用工具
        try:
            # 尝试解析 JSON（工具调用指令）
            if "{" in response_text and "}" in response_text:
                # 提取 JSON 部分
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                tool_call = json.loads(json_str)

                tool_name = tool_call.get("tool")
                tool_input = tool_call.get("input")

                if tool_name in self.tools:
                    # 5. 调用工具
                    print(f"\n[🔧 Agent 调用工具] {tool_name}('{tool_input}')")

                    tool_result = self.tools[tool_name]["func"](tool_input)
                    print(f"[✓ 工具返回] {len(tool_result)} 个字符")

                    # 6. 保存工具结果到记忆
                    self.tool_results[f"{tool_name}({tool_input})"] = tool_result

                    # 7. 记录到对话历史
                    self.conversation_history.append({
                        "role": "system",
                        "content": f"已调用工具 {tool_name} 读取文件: {tool_input}"
                    })

                    # 8. 将工具结果返回给 LLM，生成最终答案
                    follow_up_prompt = self._build_followup_prompt(
                        tool_name, tool_input, tool_result, user_input
                    )
                    final_response = self.llm.invoke([HumanMessage(content=follow_up_prompt)])
                    final_text = final_response.content

                    # 9. 保存助手回复到历史
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_text
                    })

                    return {"output": final_text}
        except Exception as e:
            print(f"[工具调用失败] {e}")

        # 如果没有调用工具，直接返回 LLM 的回复
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
        return {"output": response_text}

    def _build_prompt(self, user_input):
        """构建包含历史对话的系统提示"""
        base_prompt = f"""你是一个有帮助的助手，可以使用工具来完成任务。

{self.tool_descriptions}

当你需要使用工具时，请按以下 JSON 格式回复：
{{
    "tool": "工具名称",
    "input": "工具输入"
}}

例如：
{{
    "tool": "read_file",
    "input": "test.txt"
}}

如果不需要使用工具，直接回答用户的问题。
"""

        # 添加已读取的文件内容（记忆！）
        if "read_file(test.txt)" in self.tool_results:
            file_content = self.tool_results["read_file(test.txt)"]
            base_prompt += f"\n【已读取文件内容 - test.txt】\n{file_content}\n\n"

        # 添加对话历史（最近 10 条）
        if len(self.conversation_history) > 0:
            base_prompt += "\n【对话历史】\n"
            recent_history = self.conversation_history[-10:]
            for msg in recent_history:
                if msg["role"] == "user":
                    role = "用户"
                elif msg["role"] == "assistant":
                    role = "助手"
                else:
                    continue  # 跳过系统消息
                base_prompt += f"{role}: {msg['content']}\n"

        base_prompt += f"\n【当前用户输入】\n{user_input}\n"

        return base_prompt

    def _build_followup_prompt(self, tool_name, tool_input, tool_result, original_question):
        """构建工具调用后的后续提示"""
        return f"""工具 {tool_name}({tool_input}) 的返回结果：
---
{tool_result}
---

请基于以上结果回答用户的问题：{original_question}

注意：如果用户询问关于"内容"、"它"、"这个文件"等，都是指上面的工具返回结果。"""

    def clear_memory(self):
        """清除对话记忆和工具缓存"""
        self.conversation_history = []
        self.tool_results = {}
        print("✓ 已清除对话历史和工具缓存")
```

**核心设计思想**：

1. **对话记忆**：通过 `conversation_history` 保存所有对话
2. **工具记忆**：通过 `tool_results` 保存工具返回的结果
3. **两轮调用**：
   - 第一轮：LLM 决定是否调用工具
   - 第二轮：LLM 基于工具结果生成最终答案

### 4.3 第三步：初始化 LLM

```python
# agent.py（续）

def build_agent():
    """构建一个带记忆的 Tool-Driven Agent"""

    # 从环境变量读取配置
    api_key = os.getenv("OPENAI_API_KEY")

    # 使用智谱 API（兼容 OpenAI 接口）
    llm = ChatOpenAI(
        model="glm-4-flash",  # 智谱的模型
        temperature=0,         # 设置为 0，输出更稳定
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    # 定义工具列表
    tools = {
        "read_file": {
            "func": read_file.func,  # 使用 .func 获取实际函数
            "description": "读取文本文件的内容。输入：文件路径（str）。输出：文件内容（str）。"
        }
    }

    return SimpleAgent(llm, tools)
```

**为什么使用智谱 API？**
- ✅ 免费额度大，适合学习
- ✅ 兼容 OpenAI 接口，无需改代码
- ✅ 中文支持好
- ✅ 响应速度快

### 4.4 第四步：运行程序

```python
# main.py

from agent import build_agent

if __name__ == "__main__":
    agent = build_agent()

    print("=" * 50)
    print("  Tool-Driven Agent 已启动")
    print("  输入 'help' 查看帮助，'exit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\nUser > ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("再见！")
            break

        # 清除记忆命令
        if user_input.lower() in ["clear", "reset"]:
            agent.clear_memory()
            continue

        # 帮助命令
        if user_input.lower() == "help":
            print("""
可用命令：
  - 读取 test.txt           : 读取文件内容
  - 总结文章内容           : 总结已读取的文章
  - 文章讲了什么          : 概述文章内容
  - clear / reset          : 清除对话记忆
  - exit / quit            : 退出程序
            """)
            continue

        result = agent.invoke({"input": user_input})
        print("\nAgent >", result["output"])
```

---

## 五、运行效果演示

### 5.1 准备测试文件

创建 `test.txt`：

```txt
# 欢迎使用 Agent

这是一个示例文件，用于演示 Agent 的文件读取能力。

Agent 是一个智能体，它能够：
1. 理解用户的意图
2. 调用合适的工具
3. 处理工具返回的结果
4. 生成自然的回答
```

### 5.2 运行程序

```bash
python main.py
```

### 5.3 交互示例

**示例 1：读取文件**

```
User > 读取 test.txt 文件

[⚡ Agent 正在分析...]

[🔧 Agent 调用工具] read_file('test.txt')
[✓ 工具返回] 134 个字符

Agent > 已成功读取文件，内容如下：

# 欢迎使用 Agent

这是一个示例文件，用于演示 Agent 的文件读取能力。

Agent 是一个智能体，它能够：
1. 理解用户的意图
2. 调用合适的工具
3. 处理工具返回的结果
4. 生成自然的回答
```

**示例 2：基于已读取的内容提问**

```
User > 这篇文章讲了什么？

Agent > 这篇文章介绍了 Agent（智能体）的概念和它的四个核心能力：
1. 理解用户意图
2. 调用合适的工具
3. 处理工具返回的结果
4. 生成自然的回答
```

**示例 3：总结内容**

```
User > 请总结文章的主要内容

Agent > 这篇文章的主要内容是：
- 欢迎使用 Agent
- 介绍了 Agent 的四个核心能力
- 强调了 Agent 能够理解意图、调用工具、处理结果和生成回答
```

---

## 六、核心代码深度解析

### 6.1 为什么需要两次调用 LLM？

```python
# 第一次调用：决定是否使用工具
response = self.llm.invoke([HumanMessage(content=system_prompt)])
# 返回: {"tool": "read_file", "input": "test.txt"}

# 第二次调用：基于工具结果生成最终答案
final_response = self.llm.invoke([HumanMessage(content=follow_up_prompt)])
# 返回: "已成功读取文件，内容如下：..."
```

**原因**：
1. 第一次调用：LLM 需要理解用户意图，决定是否调用工具
2. 第二次调用：LLM 需要理解工具返回的结果，生成用户友好的回答

**这其实就是 ReAct 框架的核心**：推理 → 行动 → 观察 → 回答

### 6.2 对话历史的作用

```python
# 保存对话历史
self.conversation_history.append({
    "role": "user",
    "content": user_input
})

# 在后续对话中使用
if len(self.conversation_history) > 0:
    base_prompt += "\n【对话历史】\n"
    for msg in recent_history:
        base_prompt += f"{role}: {msg['content']}\n"
```

**为什么需要对话历史？**
- ✅ 让 Agent 能够理解上下文
- ✅ 支持多轮对话
- ✅ 可以引用之前读取的文件内容

**实际例子**：

```
User > 读取 test.txt
Agent > [读取文件]

User > 这篇文章讲了什么？  ← Agent 需要知道"这篇文章"指的是什么
Agent > [基于对话历史，知道是 test.txt，直接回答]
```

### 6.3 工具结果记忆

```python
# 保存工具结果
self.tool_results[f"{tool_name}({tool_input})"] = tool_result

# 在后续对话中使用
if "read_file(test.txt)" in self.tool_results:
    file_content = self.tool_results["read_file(test.txt)"]
    base_prompt += f"\n【已读取文件内容】\n{file_content}\n"
```

**优化**：避免重复读取同一个文件

```
User > 读取 test.txt
Agent > [调用工具读取]

User > 文件的第一行是什么？  ← 不需要再次调用工具
Agent > [直接从记忆中获取内容回答]
```

### 6.4 为什么要设置 temperature=0？

```python
llm = ChatOpenAI(
    model="glm-4-flash",
    temperature=0,  # ← 关键！
)
```

- `temperature=0`：输出更确定，适合工具调用
- `temperature=1`：输出更随机，适合创意生成

**在 Agent 中使用 temperature=0 的原因**：
1. 工具调用需要精确的 JSON 格式
2. 不需要创造性，只需要正确执行
3. 提高稳定性和可预测性

---

## 七、常见问题与解决方案

### 7.1 Agent 无法识别工具调用指令

**问题**：LLM 返回的不是 JSON 格式

**解决方案**：
```python
# 在 system prompt 中明确指定格式
"""
当你需要使用工具时，请按以下 JSON 格式回复：
{
    "tool": "工具名称",
    "input": "工具输入"
}
"""
```

### 7.2 对话历史太长怎么办？

**问题**：随着对话增多，context 越来越长

**解决方案**：
```python
# 只保留最近 N 条对话
recent_history = self.conversation_history[-10:]  # 最近 10 条
```

### 7.3 工具调用失败怎么办？

**问题**：文件不存在、权限不足等

**解决方案**：
```python
try:
    tool_result = self.tools[tool_name]["func"](tool_input)
except Exception as e:
    # 返回错误信息给 LLM
    tool_result = f"工具调用失败: {str(e)}"
```

---

## 八、扩展：添加更多工具

### 8.1 添加计算器工具

```python
# 在 tools.py 中添加

@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式

    Args:
        expression: 数学表达式，如 "2 + 3 * 4"

    Returns:
        计算结果
    """
    try:
        result = eval(expression)  # 注意：生产环境不要用 eval
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"
```

### 8.2 添加当前时间工具

```python
from datetime import datetime

@tool
def get_current_time() -> str:
    """
    获取当前日期和时间

    Returns:
        当前时间的字符串
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

### 8.3 在 Agent 中注册新工具

```python
# 在 build_agent() 函数中

tools = {
    "read_file": {
        "func": read_file.func,
        "description": "读取文本文件的内容"
    },
    "calculator": {
        "func": calculator.func,
        "description": "计算数学表达式，如 '2 + 3 * 4'"
    },
    "get_current_time": {
        "func": get_current_time.func,
        "description": "获取当前日期和时间"
    }
}
```

### 8.4 测试新工具

```
User > 现在几点了？

Agent > [调用 get_current_time 工具]
Agent > 现在的时间是 2025-01-15 14:30:25

User > 计算 123 + 456

Agent > [调用 calculator 工具]
Agent > 计算结果是 579
```

---

## 九、与 LangChain Agent 的对比

### 9.1 我们的自实现 vs LangChain

| 特性 | 我们的 SimpleAgent | LangChain Agent |
|------|-------------------|-----------------|
| 代码量 | ~150 行 | ~50 行 |
| 可控性 | 完全可控 | 封装程度高 |
| 学习价值 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 生产使用 | 需要完善 | 开箱即用 |
| 灵活性 | 高 | 中 |

### 9.2 什么时候用哪种？

**使用 SimpleAgent（自实现）**：
- ✅ 学习 Agent 原理
- ✅ 需要完全控制行为
- ✅ 简单场景

**使用 LangChain Agent**：
- ✅ 生产环境
- ✅ 复杂的工具调用
- ✅ 需要丰富的功能

---

## 十、总结

### 10.1 核心概念回顾

- ✅ **Agent**：让 LLM 能够使用工具，扩展其能力边界
- ✅ **ReAct 框架**：推理 → 行动 → 观察 → 回答
- ✅ **对话记忆**：通过保存历史实现多轮对话
- ✅ **工具记忆**：缓存工具结果，避免重复调用

### 10.2 学习清单

在继续下一篇文章之前，确保你：
- [ ] 理解 Agent 和传统 LLM 的区别
- [ ] 能够运行项目代码
- [ ] 理解 ReAct 的工作流程
- [ ] 理解两次调用 LLM 的原因
- [ ] 理解对话历史和工具记忆的作用
- [ ] 能够添加新工具

### 10.3 下一步预告

在下一篇文章《Agent 进阶：从单工具到智能助手》中，我们将深入探讨：
- 🧠 **高级 Prompt Engineering**：如何设计更好的系统提示词
- 🎯 **Few-shot Learning**：通过示例提升 Agent 准确率
- 🔧 **错误处理和重试**：让 Agent 更健壮
- 🚀 **性能优化**：减少 API 调用次数

---

## 十一、参考资料

- 📄 [智谱 AI 开放平台](https://open.bigmodel.cn/)
- 📄 [LangChain 官方文档](https://python.langchain.com/)
- 📄 [ReAct 论文原文](https://arxiv.org/pdf/2210.03629.pdf)

---

## 十二、互动与交流

如果你在实践过程中遇到问题：
- 💬 在评论区留言
- 🐛 提交 Issue
- 📧 联系我

**下一篇文章**：《Agent 进阶：从单工具到智能助手》

---

> 💡 **小贴士**：
> 1. 动手修改代码，观察 Agent 行为变化
> 2. 尝试添加新工具，扩展 Agent 能力
> 3. 调整 Prompt，看看如何提升准确率

🎉 **恭喜你完成了 Agent 开发的第一步！**
