import asyncio
import json
import logging
import os

from langchain_tavily import TavilySearch
from langchain_core.documents import Document

_cache_lock = asyncio.Lock()

web_search_tool = TavilySearch(max_results=3)

logger = logging.getLogger(__name__)

TAVILY_CACHE_PATH = "data/tavily_search_results.json"


def _load_tavily_cache() -> dict:
    if os.path.exists(TAVILY_CACHE_PATH):
        try:
            with open(TAVILY_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(
                "Error loading Tavily cache: %s. Starting with empty cache.", e
            )
            return {}
    return {}


def _save_tavily_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(TAVILY_CACHE_PATH), exist_ok=True)
    with open(TAVILY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)


async def fetch_web_documents(query: str) -> list[Document]:
    """Tavily search with file cache. Shared by the CRAG node and the agent tool."""
    if not query or not query.strip():
        return []

    async with _cache_lock:
        cache = _load_tavily_cache()

    if query in cache:
        logger.info("Tavily cache hit: %s", query)
        results = cache[query]
    else:
        logger.warning("Tavily cache miss, searching web: %s", query)
        web_results = await web_search_tool.ainvoke(query)
        results = web_results.get("results", [])
        async with _cache_lock:
            cache = _load_tavily_cache()
            cache[query] = results
            _save_tavily_cache(cache)

    docs = []
    for r in results:
        content = r.get("content", "").strip()
        if content:
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": r.get("url") or "Web Search"},
                )
            )
    return docs
