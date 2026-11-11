import logging
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.tools import search_docs, search_web
from src.agent.agent_state import AgentState
from src.agent.nodes import create_agent_node
from src.config import DEFAULT_LLM

logger = logging.getLogger(__name__)


def agent_pipeline(llm=DEFAULT_LLM):
    """An agent pipeline that uses a state graph to manage the flow of information between the agent and tools."""
    tool_node = ToolNode([search_docs, search_web])

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", create_agent_node(llm=llm))
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()