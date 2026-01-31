# tools.py
from langchain.tools import tool
import os

@tool
def read_file(path: str) -> str:
    """
    Read the content of a text file.
    """
    if not os.path.exists(path):
        return "File not found"

    with open(path, "r", encoding="utf-8") as f:
        return f.read()
