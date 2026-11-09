from langgraph.prebuilt import ToolNode
from src.agent.tools import search_docs, search_web
from src.agent.agent_prompt import prompt
from src.agent.agent_state import AgentState
from src.config import DEFAULT_LLM

def create_agent_node(llm=DEFAULT_LLM):
    model_with_tools = llm.bind_tools([search_docs, search_web])
    chain = prompt | model_with_tools

    async def agent_node(state: AgentState):
        messages = state.get("messages", [])
        selected_files = state.get("selected_files")

        response = await chain.ainvoke({
            "messages": messages,
            "selected_files": selected_files or "None",
        })

        return {"messages": [response]}

    return agent_node