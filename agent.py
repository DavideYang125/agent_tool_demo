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
    """构建一个简单的 Tool-Driven Agent"""

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
    """简单的 Tool-Driven Agent 实现"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.tool_descriptions = self._format_tool_descriptions()

    def _format_tool_descriptions(self):
        """格式化工具描述"""
        desc = "\n可用工具：\n"
        for name, tool_info in self.tools.items():
            desc += f"- {name}: {tool_info['description']}\n"
        return desc

    def invoke(self, inputs):
        """处理用户输入"""
        user_input = inputs.get("input", "")

        # 系统提示
        system_prompt = f"""你是一个有帮助的助手，可以使用工具来完成任务。

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

用户输入：{user_input}
"""

        # 第一次调用 LLM
        response = self.llm.invoke([HumanMessage(content=system_prompt)])
        response_text = response.content

        print(f"\n[Agent 思考中] {response_text}")

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
                    print(f"\n[Agent 调用工具] {tool_name}({tool_input})")
                    tool_result = self.tools[tool_name]["func"](tool_input)
                    print(f"[工具返回] {tool_result}")

                    # 将工具结果返回给 LLM
                    follow_up_prompt = f"""工具 {tool_name} 的返回结果：
{tool_result}

请基于以上结果回答用户的问题。"""
                    final_response = self.llm.invoke([HumanMessage(content=follow_up_prompt)])
                    return {"output": final_response.content}
        except Exception as e:
            print(f"[工具调用失败] {e}")

        # 如果没有调用工具，直接返回
        return {"output": response_text}
