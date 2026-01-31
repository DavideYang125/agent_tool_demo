# main.py
from agent import build_agent

if __name__ == "__main__":
    agent = build_agent()

    while True:
        user_input = input("\nUser > ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = agent.invoke(
            {"input": user_input}
        )

        print("\nAgent >", result["output"])
