# Architectural Decisions & Benchmark History

> **Note:** This document captures key architectural decisions across all iterations and archives earlier experimental phases—including baseline benchmarks run on models since deprecated by OpenAI.

---

## 1. Retrieval Layer Evaluation: Standard Chunking (V1) vs. Contextual Chunking (V2)

- **Dataset:** 50 Evaluation Questions across 3 Academic Papers (arxiv1, arxiv2, arxiv3).
- **Techniques Evaluated:**
  - **Dense:** Vector search with `text-embedding-3-small` in ChromaDB.
  - **BM25s:** Sparse lexical search with Porter stemmer.
  - **Hybrid:** Reciprocal Rank Fusion (RRF) combining Dense and BM25 candidate pools ($k=20$).
  - **Reranker:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) scoring fused candidates down to top-3 or top-5 chunks.

| Experiment ID | Retrieval Strategy | V1 Hit Rate | V1 MRR | V2 (Contextual) Hit Rate | V2 MRR | V2 Avg Latency | V2 P95 Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp A (k=3)** | Dense Vector Search | 0.73 | 0.58 | 0.72 | 0.55 | 1,003 ms | 2,530 ms |
| **Exp A (k=5)** | Dense Vector Search | 0.88 | 0.62 | 0.88 | 0.58 | 1,052 ms | 2,778 ms |
| **Exp B (k=3)** | Sparse BM25 Search | 0.81 | 0.69 | 0.74 | 0.62 | 828 ms | 2,221 ms |
| **Exp C (k=5)** | Hybrid (Dense + BM25 + RRF) | **0.88** | **0.66** | **0.88** | **0.66** | 1,254 ms | 2,744 ms |
| **Exp D (top=3)** | Hybrid + Cross-Encoder Rerank | 0.85 | 0.69 | 0.80 | 0.65 | 8,010 ms | 10,302 ms |
| **Exp E (top=5)** | Hybrid + Cross-Encoder Rerank | **0.92** | **0.71** | 0.86 | 0.64 | 7,855 ms | 10,010 ms |

### Key Retrieval Takeaways
1. **Hybrid Retrieval (Exp C & E):** Combining dense semantic search with sparse BM25 significantly outperforms either standalone approach, pushing Hit Rate to **0.92** and MRR to **0.71**.
2. **Contextual Retrieval Trade-off:** While enriching chunks with whole-document context increases chunk clarity, cross-encoder reranking latency jumps to ~8.0s on local CPU inference.

**Why I Switched to Contextual Retrieval:** Standard chunking strips away surrounding document context—if a chunk is split mid-section or mid-thought, the retriever and LLM lose track of its overarching topic. Contextual Retrieval prepends a brief, document-aware summary to each chunk during ingestion. Although this introduces a minor upfront LLM cost during indexing, it yields significantly more coherent, self-contained chunks at query time.

---

## 2. Generation Layer Evaluation: V1 (LCEL Baseline) vs. V2 (LangGraph Async)

- **Dataset:** 60 Benchmark Questions (50 Document-grounded + 10 Out-of-Scope; 20 Multi-turn follow-ups).
- **Independent Judge:** LLM Judge running structured evaluations on a 0–5 scale for **Faithfulness** (absence of hallucination) and **Relevance** (alignment with the user query).
- **Execution:** Async concurrent batch execution with semaphore flow control (`max_concurrency=5`).

| Model Name | Architectural Stage | Temperature | Faithfulness (0-5) | Relevance (0-5) | Multi-turn Score | Avg Generation Latency | P95 Latency | Avg Cost / Query | Total Cost (60 Queries) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **GPT-4o-mini (Deprecated)** | V1 LCEL Baseline | 0.0 | **4.93** | **4.95** | **5.00** | **4.64 s** | 7.61 s | $0.00018 | $0.0108 |
| **GPT-4o-mini (Deprecated)** | V2 LangGraph Async | 0.0 | **4.93** | **4.95** | **5.00** | **4.64 s** | 7.36 s | $0.00018 | $0.0110 |
| **GPT-5-nano (Deprecated)** | V2 LangGraph Async | 0.0 | 4.67 | 4.53 | 5.00 | 11.57 s | 21.73 s | $0.00056 | $0.0337 |
| **GPT-6-luna (Early Trial)** | V2 LangGraph Async | 1.0 *(Reasoning On)* | 4.70 | 4.95 | 4.80 | 7.71 s | 11.63 s | $0.00016 | $0.0095 |

### Key Generation Takeaways
1. **Architectural Parity:** Moving from LCEL to LangGraph Async maintained identical accuracy (4.93/4.95) and latency (4.64s), while unlocking stateful graph control, checkpointing, and cyclic tool loops required for Agentic RAG.
2. **Deterministic Adherence:** `gpt-4o-mini (Deprecated)` demonstrated superior context adherence at `temp=0.0`. In contrast, reasoning models running with unconstrained thinking (e.g., `gpt-5-nano` and early `gpt-6-luna` at `temp=1.0`) suffered from extrapolation and higher latency.

**Why I Switched from LangChain LCEL to LangGraph:** What started as a learning project using standard LCEL quickly hit architectural limits as requirements grew. Linear LCEL chains are rigid for complex RAG workflows. Transitioning to **LangGraph** introduced stateful orchestration with built-in SQLite checkpointing between nodes, clean error isolation, and native support for cyclic flows. Most importantly, introducing autonomous agent loops or conditional fallback routing would have been cumbersome in pure LCEL, whereas LangGraph makes multi-mode execution (`CRAG`, `Agent`, `Escalate`) modular, maintainable, and easy to inspect.

---

## 3. Advanced Architectures (V3, V4, V5) — Legacy Reference (`GPT-4o-mini (Deprecated)`)

- **Dataset:** 60 Benchmark Questions (50 Document-grounded + 10 Out-of-Scope; 20 Multi-turn follow-ups).
- **Model Configuration:** `gpt-4o-mini (Deprecated)` at `temperature=0.0`.

| Architecture / Mode | Faithfulness (0-5) | Relevance (0-5) | Multi-turn Score | Avg Gen Latency | P95 Gen Latency | Avg Tokens / Query | Avg Cost / Query | Total Cost (60 Queries) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V3 CRAG (Single-Pass)** | **5.00** | **5.00** | **5.00** | **6.83 s** | **11.33 s** | 2,523 | $0.00042 | $0.0253 |
| **V4 Pure ReAct Agent** | **5.00** | **5.00** | **5.00** | 7.64 s | 15.54 s | 2,417 | $0.00046 | $0.0276 |
| **V5 Adaptive Escalate** | **4.93** *(59/60: 5.0)* | **4.93** | **5.00** | 8.22 s | 18.20 s | 2,744 | $0.00047 | $0.0283 |

**Why I Added CRAG:** CRAG introduces a crucial quality-control gate (the Document Grader) between the retriever and the generator. In standard RAG, if the retriever fetches irrelevant chunks, the LLM either refuses to answer or hallucinates. CRAG prevents this by evaluating chunk relevance upfront: if retrieved passages fail the check, it rewrites the query and falls back to Tavily web search to find grounded information. If standard RAG operates blindly on whatever the retriever returns, CRAG acts as a verification layer that ensures only relevant evidence reaches generation—a benefit clearly reflected in the faithfulness scores.

**Why I Added the ReAct Agent:** Pure ReAct agents are inherently more flexible than static single-pass pipelines. Through iterative thought-action-observation loops, the agent can decompose multi-hop questions, query different documents selectively, and synthesize stronger answers when a single retrieval pass falls short. However, autonomy introduces risks: higher token consumption, potential loop latency, and a non-zero risk of straying if left unconstrained. To control cost and latency, strict recursion limits (`recursion_limit=10`) and boundary-aware system prompts were enforced—which ultimately motivated building the hybrid Escalate mode.

**Why I Added Adaptive Escalate Mode:** In pure CRAG, when the grader rejects retrieved chunks, the pipeline immediately jumps to web search—even if a better-phrased second lookup inside the user's documents could have found the answer. Adaptive Escalate solves this by handing control over to a `fallback_agent` whenever CRAG grading fails. The agent can re-query the document store with refined terms or choose web search only when genuinely needed.

While all three modes perform strongly, **CRAG** remains the most cost- and latency-optimized choice for large enterprises handling high volumes of straightforward document lookups. **Adaptive Escalate**, on the other hand, is recommended when maximizing answer accuracy and recovery rate is worth a modest increase in latency and token cost.

---

## 4. Advanced Architectures (V3, V4, V5) — Current Production (`GPT-6-luna`)

- **Model Configuration:** `gpt-6-luna` (`temperature=0.0`, `reasoning_effort="none"`).
- **Pricing Tier:** Ultra-low cost ($0.10 / 1M Input, $0.50 / 1M Output).

| Architecture / Mode                 | Configuration              | Faithfulness (0-5) | Relevance (0-5) | Multi-turn Score | Avg Gen Latency | P95 Gen Latency | Avg Tokens / Query | Avg Cost / Query | Total Cost (60 Queries) |
| ----------------------------------- | -------------------------- | :----------------: | :-------------: | :--------------: | :-------------: | :-------------: | :----------------: | :--------------: | :---------------------: |
| **V3 CRAG (Single-Pass)**           | `temp=0.0, reasoning=none` |        4.72        |      4.93       |     **4.95**     |   **6.94 s**    |   **10.53 s**   |       2,369        |   **$0.00027**   |       **$0.0164**       |
| **V5 Adaptive Escalate**            | `temp=0.0, reasoning=none` | **4.81** *(+0.09)* |      4.95       |     **4.95**     |   **7.23 s**    |     15.50 s     |       2,904        |   **$0.00034**   |       **$0.0205**       |
| **V4 Pure ReAct Agent**             | `temp=0.0, reasoning=none` |      **4.90**      |    **4.98**     |       4.85       |     9.98 s      |     35.21 s     |       3,472        |   **$0.00046**   |       **$0.0279**       |
| *(Trial)* **V3 CRAG (Thinking On)** | `temp=1.0, reasoning=open` |        4.87        |      4.97       |       4.85       |     10.52 s     |     16.21 s     |       2,944        |     $0.00033     |         $0.0200         |

**Model Decisions:** All evaluations in this project focused on OpenAI models. Empirically, **non-reasoning configurations (`reasoning_effort="none"`, `temperature=0.0`) consistently outperformed unconstrained reasoning models** in strict RAG settings. Reasoning-enabled models (such as `gpt-5-mini`, `gpt-5-nano`, or `gpt-6-luna` at its default `temperature=1.0`) tend to suffer from **"over-inference" (being overly clever)**—extrapolating beyond the retrieved text by injecting outside domain knowledge (e.g., adding `"(unless P=NP)"` when the source text only states `"NP-complete"`). Under strict LLM-as-a-Judge faithfulness rubrics, any unsupported extrapolation is penalized as a hallucination. By disabling internal reasoning (`reasoning_effort="none"`) on **`gpt-6-luna`**, we unlocked strict `temperature=0.0` determinism, full tool-calling compatibility, zero extrapolation, and an ultra-low pricing tier (**$0.10 / $0.50 per 1M tokens**).
