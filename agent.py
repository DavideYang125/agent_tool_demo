# agent.py
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

# 加载环境变量
load_dotenv()
from tools import read_file

def build_agent():
    """构建一个带记忆的 Tool-Driven Agent"""

    # 从环境变量读取配置
    api_key = os.getenv("OPENAI_API_KEY")

    # 使用智谱 API
    llm = ChatOpenAI(
        model="glm-4-flash",
        temperature=0,
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


class SimpleAgent:
    """带记忆的 Tool-Driven Agent 实现"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.tool_descriptions = self._format_tool_descriptions()
        # 对话历史
        self.conversation_history = []
        # 工具调用记录（用于后续引用）
        self.tool_results = {}

    def clear_memory(self):
        """清除对话记忆和工具缓存"""
        self.conversation_history = []
        self.tool_results = {}
        print("✓ 已清除对话历史和工具缓存")

    def _format_tool_descriptions(self):
        """格式化工具描述"""
        desc = "\n可用工具：\n"
        for name, tool_info in self.tools.items():
            desc += f"- {name}: {tool_info['description']}\n"
        return desc

    def invoke(self, inputs):
        """处理用户输入（带记忆）"""
        user_input = inputs.get("input", "")

        # 添加用户消息到历史
        self.conversation_history.append({"role": "user", "content": user_input})

        # 构建系统提示（包含对话历史）
        system_prompt = self._build_prompt(user_input)

        # 第一次调用 LLM（移除详细的调试输出，加快速度）
        response = self.llm.invoke([HumanMessage(content=system_prompt)])
        response_text = response.content

        # 只在需要时显示思考过程
        if "tool" in user_input.lower() or "读取" in user_input or "read" in user_input.lower():
            print(f"\n[⚡ Agent 正在分析...]")

        # 检查是否需要调用工具
        try:
            # 尝试解析 JSON
            if "{" in response_text and "}" in response_text:
                # 提取 JSON 部分
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                tool_call = json.loads(json_str)

                tool_name = tool_call.get("tool")
                tool_input = tool_call.get("input")

                if tool_name in self.tools:
                    # 调用工具
                    print(f"\n[📄 Agent 调用工具] {tool_name}('{tool_input}')")

                    tool_result = self.tools[tool_name]["func"](tool_input)
                    print(f"[✓ 工具返回] {len(tool_result)} 个字符")

                    # 保存工具结果到记忆
                    self.tool_results[f"{tool_name}({tool_input})"] = tool_result

                    # 在对话历史中标记已读取文件（不保存完整内容，避免过长）
                    self.conversation_history.append({
                        "role": "system",
                        "content": f"[系统] 已成功调用工具 {tool_name} 读取文件: {tool_input}，文件内容已保存到工作记忆中。"
                    })

                    # 将工具结果返回给 LLM
                    follow_up_prompt = self._build_followup_prompt(tool_name, tool_input, tool_result, user_input)
                    final_response = self.llm.invoke([HumanMessage(content=follow_up_prompt)])
                    final_text = final_response.content

                    # 添加助手回复到历史
                    self.conversation_history.append({"role": "assistant", "content": final_text})

                    return {"output": final_text}
        except Exception as e:
            if self._should_debug():
                print(f"[工具调用失败] {e}")

        # 如果没有调用工具，直接返回
        self.conversation_history.append({"role": "assistant", "content": response_text})
        return {"output": response_text}

    def _should_debug(self):
        """判断是否显示调试信息"""
        import os
        return os.getenv("DEBUG", "false").lower() == "true"

    def _build_prompt(self, user_input):
        """构建包含历史对话的提示"""
        # 基础系统提示
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

【重要】如果用户询问"文章"、"文件"、"内容"等，且之前已经读取过 test.txt 文件，请基于对话历史中已保存的内容回答。

"""

        # 添加已读取的文件内容到上下文
        if "read_file(test.txt)" in self.tool_results:
            file_content = self.tool_results["read_file(test.txt)"]
            base_prompt += f"\n【已读取文件内容 - test.txt】\n{file_content}\n\n"

        # 添加对话历史（最近 5 轮，不包含系统消息）
        if len(self.conversation_history) > 0:
            base_prompt += "\n【对话历史】\n"
            recent_history = self.conversation_history[-10:]  # 保留最近 10 条
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
