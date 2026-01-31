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
            print("✓ 对话记忆已清除")
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
