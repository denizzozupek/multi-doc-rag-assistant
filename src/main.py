import argparse
import asyncio
import logging
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.config import DEFAULT_LLM, DEFAULT_RERANK_TOP_K
from src.graph.crag import graph_pipeline
from src.agent.pipeline import agent_pipeline
from src.graph.escalate import escalate_graph_pipeline

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_pipeline(
    mode: str = "crag",
    llm: BaseChatModel = DEFAULT_LLM,
    top_k: int = DEFAULT_RERANK_TOP_K,
    crag_enable: bool = True,
) -> Any:
    """Factory function that returns the compiled pipeline app for the specified mode.
    
    Modes:
      - 'crag': Fast, single-pass deterministic CRAG workflow (Default).
      - 'agent': Autonomous multi-hop ReAct agent with tools.
      - 'escalate': Adaptive progressive escalation (CRAG first, escalates to Agent on failure).
    """
    mode = mode.lower().strip()
    if mode == "crag":
        return graph_pipeline(llm=llm, top_k=top_k, crag_enable=crag_enable).compile()
    elif mode == "agent":
        return agent_pipeline(llm=llm)
    elif mode == "escalate":
        return escalate_graph_pipeline(llm=llm, top_k=top_k, crag_enable=crag_enable).compile()
    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported modes: 'crag', 'agent', 'escalate'.")


async def ask_question(
    app: Any,
    query: str,
    selected_files: list[str] | None = None,
    recursion_limit: int = 10,
    stream: bool = False,
) -> str:
    """Asynchronously runs a user query through the compiled pipeline with recursion protection."""
    state_input = {
        "messages": [HumanMessage(content=query)],
        "selected_files": selected_files,
    }
    config = {"recursion_limit": recursion_limit}

    if stream:
        print(f"\n--- [Running Query: {query}] ---")
        last_msg = ""
        async for event in app.astream(state_input, config=config):
            for node_name, output in event.items():
                print(f" Node Executed: {node_name}")
                if "messages" in output and output["messages"]:
                    msg = output["messages"][-1]
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"Tool Call: {tc['name']}({tc['args']})")
                    elif msg.type == "tool":
                        preview = str(msg.content)[:120].replace("\n", " ")
                        print(f"Tool Observation: {preview}...")
                    else:
                        last_msg = msg.content
        return last_msg
    else:
        output = await app.ainvoke(state_input, config=config)
        return output["messages"][-1].content


async def interactive_chat(app: Any, selected_files: list[str] | None = None, mode: str = "crag"):
    """Runs a continuous interactive CLI chat loop."""
    print(f"\n=======================================================")
    print(f" Multi-Doc RAG Assistant Active [Mode: {mode.upper()}]")
    print(f" Selected Files: {', '.join(selected_files) if selected_files else 'All available docs'}")
    print(f" Type 'exit' or 'quit' to terminate.")
    print(f"=======================================================\n")

    while True:
        try:
            query = input("User > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            response = await ask_question(
                app=app,
                query=query,
                selected_files=selected_files,
                stream=True,
            )
            print(f"\nAssistant > {response}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Doc RAG Assistant Entrypoint (CRAG, Agent, Escalate)"
    )
    parser.add_argument(
        "--mode",
        choices=["crag", "agent", "escalate"],
        default="crag",
        help="Pipeline mode to execute (default: crag)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional single query to evaluate. If omitted, starts interactive chat.",
    )
    parser.add_argument(
        "--files",
        type=str,
        default=None,
        help="Comma-separated filenames to restrict context (e.g. 'arxiv1.pdf,arxiv2.pdf')",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream step-by-step node executions to stdout",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_files = [f.strip() for f in args.files.split(",")] if args.files else None

    app = get_pipeline(mode=args.mode)

    if args.query:
        answer = asyncio.run(
            ask_question(
                app=app,
                query=args.query,
                selected_files=selected_files,
                stream=args.stream,
            )
        )
        print(f"\nResult:\n{answer}")
    else:
        asyncio.run(
            interactive_chat(
                app=app,
                selected_files=selected_files,
                mode=args.mode,
            )
        )
