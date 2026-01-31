# agent.py - 带详细注释的版本
"""
Agent 核心实现文件

核心功能：
1. 初始化 LLM 和工具
2. 管理对话历史和工具结果缓存
3. 实现 ReAct 循环：思考 → 行动 → 观察
4. 动态构建 Prompt
"""

import os
import sys
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 设置 UTF-8 编码（Windows 兼容）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# 加载环境变量（从 .env 文件读取 API Key）
load_dotenv()
from tools import read_file


def build_agent():
    """
    构建 Agent 的工厂函数

    返回：
        SimpleAgent 实例
    """
    # 从环境变量读取 API Key
    api_key = os.getenv("OPENAI_API_KEY")

    # 初始化 LLM（大语言模型）
    # 使用智谱 API，兼容 OpenAI 格式
    llm = ChatOpenAI(
        model="glm-4-flash",        # 模型名称（智谱的免费模型）
        temperature=0,              # 温度=0：输出更确定，适合工具调用
        api_key=api_key,            # API 密钥
        base_url="https://open.bigmodel.cn/api/paas/v4/"  # API 端点
    )

    # 定义工具列表
    # 工具是 Agent 与外部世界交互的接口
    tools = {
        "read_file": {
            "func": read_file.func,  # .func 获取实际的可调用函数
            "description": "读取文本文件的内容。输入：文件路径（str）。输出：文件内容（str）。"
        }
    }

    # 创建并返回 Agent 实例
    return SimpleAgent(llm, tools)


class SimpleAgent:
    """
    简单的 Tool-Driven Agent 实现

    属性：
        llm: 大语言模型实例
        tools: 可用工具字典
        conversation_history: 对话历史列表
        tool_results: 工具执行结果缓存

    核心方法：
        invoke(): 处理用户输入的主方法
        _build_prompt(): 构建包含上下文的 Prompt
        _build_followup_prompt(): 构建工具调用后的 Prompt
        clear_memory(): 清除对话记忆
    """

    def __init__(self, llm, tools):
        """
        初始化 Agent

        参数：
            llm: ChatOpenAI 实例
            tools: 工具字典 {"tool_name": {"func": callable, "description": str}}
        """
        self.llm = llm                    # LLM 是 Agent 的大脑
        self.tools = tools                 # 工具是 Agent 的手
        self.tool_descriptions = self._format_tool_descriptions()

        # 对话历史：存储用户和助手的历史对话
        # 格式：[{"role": "user|assistant", "content": "对话内容"}]
        self.conversation_history = []

        # 工具结果缓存：避免重复调用相同工具
        # 格式：{"tool_name(input)": "result"}
        self.tool_results = {}

    def clear_memory(self):
        """清除对话记忆和工具缓存"""
        self.conversation_history = []
        self.tool_results = {}
        print("✓ 已清除对话历史和工具缓存")

    def _format_tool_descriptions(self):
        """
        格式化工具描述，用于 Prompt

        返回：
            str: 工具描述文本
        """
        desc = "\n可用工具：\n"
        for name, tool_info in self.tools.items():
            desc += f"- {name}: {tool_info['description']}\n"
        return desc

    def invoke(self, inputs):
        """
        处理用户输入的主方法（核心逻辑）

        这是 Agent 的大脑，实现完整的 ReAct 循环：
        1. 理解用户输入
        2. 决定是否需要工具
        3. 如果需要，调用工具
        4. 基于工具结果生成回答

        参数：
            inputs: {"input": "用户输入"}

        返回：
            {"output": "Agent 的回答"}
        """
        # ========== 步骤1：记录用户输入 ==========
        user_input = inputs.get("input", "")
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # ========== 步骤2：构建 Prompt ==========
        # Prompt 包含：
        # - 系统指令
        # - 工具描述
        # - 已读取的文件内容
        # - 对话历史
        # - 当前用户输入
        system_prompt = self._build_prompt(user_input)

        # ========== 步骤3：LLM 第一次推理 ==========
        # LLM 分析用户输入，决定是否需要调用工具
        response = self.llm.invoke([HumanMessage(content=system_prompt)])
        response_text = response.content

        # 调试信息（只在需要时显示）
        if "tool" in user_input.lower() or "读取" in user_input or "read" in user_input.lower():
            print(f"\n[⚡ Agent 正在分析...]")

        # ========== 步骤4：检查是否需要工具 ==========
        try:
            # 尝试从 LLM 响应中解析 JSON 格式的工具调用
            if "{" in response_text and "}" in response_text:
                # 提取 JSON 部分（处理可能的额外文本）
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                tool_call = json.loads(json_str)

                tool_name = tool_call.get("tool")
                tool_input = tool_call.get("input")

                # ========== 步骤5：调用工具 ==========
                if tool_name in self.tools:
                    print(f"\n[📄 Agent 调用工具] {tool_name}('{tool_input}')")

                    # 执行工具函数
                    tool_result = self.tools[tool_name]["func"](tool_input)
                    print(f"[✓ 工具返回] {len(tool_result)} 个字符")

                    # 缓存工具结果（后续可以重用）
                    self.tool_results[f"{tool_name}({tool_input})"] = tool_result

                    # 在对话历史中记录工具调用
                    self.conversation_history.append({
                        "role": "system",
                        "content": f"[系统] 已成功调用工具 {tool_name} 读取文件: {tool_input}，文件内容已保存到工作记忆中。"
                    })

                    # ========== 步骤6：将工具结果返回给 LLM ==========
                    # 构建新的 Prompt，包含工具返回的结果
                    follow_up_prompt = self._build_followup_prompt(
                        tool_name, tool_input, tool_result, user_input
                    )
                    final_response = self.llm.invoke([HumanMessage(content=follow_up_prompt)])
                    final_text = final_response.content

                    # 记录助手回复
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_text
                    })

                    return {"output": final_text}

        except Exception as e:
            # 工具调用失败处理
            if self._should_debug():
                print(f"[工具调用失败] {e}")

        # ========== 步骤7：直接返回 LLM 回答（无需工具） ==========
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
        return {"output": response_text}

    def _should_debug(self):
        """判断是否显示调试信息（从环境变量读取）"""
        import os
        return os.getenv("DEBUG", "false").lower() == "true"

    def _build_prompt(self, user_input):
        """
        构建包含历史对话的 Prompt

        这是 Prompt Engineering 的核心！
        好的 Prompt 能够显著提升 Agent 的性能。

        参数：
            user_input: 用户输入

        返回：
            str: 完整的 Prompt
        """
        # ========== 基础系统提示 ==========
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

【重要】如果用户询问"文章"、"文件"、"内容"等，且之前已经读取过 test.txt 文件，
请基于对话历史中已保存的内容回答。

"""

        # ========== 关键优化：注入已读取的文件内容 ==========
        # 这解决了"为什么 Agent 记不住文件内容"的问题
        if "read_file(test.txt)" in self.tool_results:
            file_content = self.tool_results["read_file(test.txt)"]
            base_prompt += f"\n【已读取文件内容 - test.txt】\n{file_content}\n\n"

        # ========== 添加对话历史 ==========
        # 只保留最近 10 轮对话，避免 Prompt 过长
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

        # ========== 当前用户输入 ==========
        base_prompt += f"\n【当前用户输入】\n{user_input}\n"

        return base_prompt

    def _build_followup_prompt(self, tool_name, tool_input, tool_result, original_question):
        """
        构建工具调用后的后续 Prompt

        参数：
            tool_name: 工具名称
            tool_input: 工具输入
            tool_result: 工具返回结果
            original_question: 用户原始问题

        返回：
            str: Prompt
        """
        return f"""工具 {tool_name}({tool_input}) 的返回结果：
---
{tool_result}
---

请基于以上结果回答用户的问题：{original_question}

注意：如果用户询问关于"内容"、"它"、"这个文件"等，都是指上面的工具返回结果。"""


"""
使用示例：

# 创建 Agent
agent = build_agent()

# 处理用户输入
result = agent.invoke({"input": "读取test.txt"})
print(result["output"])

# 基于已读取的内容提问
result = agent.invoke({"input": "文章讲了什么？"})
print(result["output"])

# 清除记忆
agent.clear_memory()
"""
