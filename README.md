# Tool-Driven 单 Agent 最小可运行版本

这是一个最小可运行的 Tool-Driven Agent 示例，演示了完整的流程：
**LLM → 决策 → Tool 调用 → 返回结果**

## 项目结构

```
agent_tool_demo/
├── agent.py          # Agent 配置
├── tools.py          # 工具定义
├── main.py           # 主程序入口
├── test.txt          # 测试文件
├── requirements.txt  # 依赖列表
├── .env              # 环境变量配置
└── .env.example      # 环境变量示例
```

## 快速开始

### 1. 安装依赖

使用虚拟环境（推荐）：
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

**使用智谱 AI（推荐，有免费额度）：**

编辑 `.env` 文件，填入智谱 API Key：

```bash
OPENAI_API_KEY=your_zhipu_api_key_here
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
```

获取智谱 API Key：
- 访问 https://open.bigmodel.cn/usercenter/apikeys
- 注册/登录后创建新的 API Key
- 智谱提供免费额度，适合测试使用

**使用 OpenAI（需要付费）：**

```bash
OPENAI_API_KEY=sk-your-openai-key-here
# OPENAI_API_BASE=https://api.openai.com/v1  # 可选，默认值
```

### 3. 运行程序

```bash
python main.py
```

然后可以尝试以下对话：
```
User > 请读取 test.txt 文件的内容
User > exit
```

## 代码说明

### tools.py - 工具定义
使用 LangChain 的 `@tool` 装饰器定义工具：
```python
@tool
def read_file(path: str) -> str:
    """
    Read the content of a text file.
    """
    # 读取文件逻辑
```

### agent.py - Agent 配置
- 创建 LLM 模型
- 定义工具列表
- 使用 `create_agent` 创建 Agent

### main.py - 主程序
- 构建 Agent
- 交互式对话循环

## 依赖说明

- `langchain>=0.1.0` - LangChain 核心库
- `langchain-openai>=0.1.0` - OpenAI 集成
- `langchainhub>=0.1.0` - LangChain Hub（Prompt 模板）
- `python-dotenv` - 环境变量管理

## 技术栈

- **LLM**: 智谱 GLM-4-Flash（默认）/ OpenAI GPT-4o-mini
- **Framework**: LangChain 1.2.7
- **Agent Pattern**: Tool-Calling Agent
- **Architecture**: ReAct (Reasoning + Acting)

## 常见问题

### Q: 如何使用其他 LLM？
A: 修改 `agent.py` 中的 `ChatOpenAI` 配置，或替换为其他 LLM 提供商。

### Q: 如何添加更多工具？
A: 在 `tools.py` 中使用 `@tool` 装饰器添加新函数，然后在 `agent.py` 中加入 tools 列表。

### Q: 支持哪些文件格式？
A: 当前 `read_file` 工具只支持文本文件。可以扩展工具支持更多格式。
