"""
agent_demo.py — A minimal terminal chat agent that uses AgentMemory
(FlyHash-backed) as a tool, via the Claude API.

Setup:
    pip install anthropic
    setx ANTHROPIC_API_KEY "your-key-here"      (Windows, new terminal after)
    export ANTHROPIC_API_KEY="your-key-here"    (macOS/Linux)

Run:
    python agent_demo.py

Try a flow like:
    You: My favorite language is Python and I'm allergic to peanuts.
    You: (new topic) What's a good weekend project idea?
    You: What should I avoid eating?
    -> the agent should call recall_memory and correctly answer "peanuts"
       even though that fact was mentioned turns ago, not in this message.

This is intentionally small — the point is demonstrating the memory tool
integration, not building a production agent framework.
"""

from __future__ import annotations
import os
import sys
import json

from agent_memory import AgentMemory

try:
    import anthropic
except ImportError:
    print("Run: pip install anthropic")
    sys.exit(1)


TOOLS = [
    {
        "name": "save_memory",
        "description": (
            "Save a fact worth remembering for future conversations "
            "(user preferences, constraints, personal details they share)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The fact to remember, as a standalone sentence."}
            },
            "required": ["fact"],
        },
    },
    {
        "name": "recall_memory",
        "description": (
            "Search remembered facts for anything relevant to the current "
            "question. Use this whenever the user asks something that might "
            "depend on something they told you earlier in a past turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search memory for."}
            },
            "required": ["query"],
        },
    },
]

SYSTEM_PROMPT = (
    "You are a helpful assistant with long-term memory. When the user "
    "shares a personal fact, preference, or constraint, call save_memory. "
    "When answering a question that might depend on something said earlier "
    "in the conversation, call recall_memory first, then answer using what "
    "it returns. Don't mention the tools by name to the user."
)


def run_tool(memory: AgentMemory, name: str, tool_input: dict) -> str:
    if name == "save_memory":
        memory.remember(tool_input["fact"])
        return f"Saved: {tool_input['fact']}"
    elif name == "recall_memory":
        return memory.recall_as_context(tool_input["query"], k=3)
    return "Unknown tool"


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY as an environment variable first, then rerun.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    memory = AgentMemory()
    print(f"\nAgent ready. {len(memory)} memories loaded. Type 'quit' to exit.\n")

    conversation = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        conversation.append({"role": "user", "content": user_input})

        # Loop until the model stops requesting tool calls.
        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=conversation,
            )

            conversation.append({"role": "assistant", "content": response.content})

            tool_calls = [b for b in response.content if b.type == "tool_use"]
            if not tool_calls:
                text_blocks = [b.text for b in response.content if b.type == "text"]
                print("Agent:", " ".join(text_blocks))
                break

            tool_results = []
            for call in tool_calls:
                result = run_tool(memory, call.name, call.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result,
                })
            conversation.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    main()
