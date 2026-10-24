import json
from pathlib import Path

results_path = Path("eval/evaluation_results")

for file in results_path.glob("Exp_*.json"):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = file.stem

    for detailed_result in data.get("detailed_results", []):
        if detailed_result.get("hit_score") == 0:
            retrieved_docs = [
                doc.get("content", "")[:120]
                for doc in detailed_result.get("retrieved_docs", [])
            ]
            print(f"Missed retrieval for {filename}:")
            print(f"  Raw Query: {detailed_result.get('raw_query')}")
            print(f"  Search Query: {detailed_result.get('search_query')}")
            print(f"  Expected Text Match: {detailed_result.get('expected_text_match')}")
            print(f"  Top K: {detailed_result.get('top_k')}")
            print("  Retrieved Documents:")
            for rank, doc in enumerate(retrieved_docs, start=1):
                print(f"    Rank {rank}: {doc}")
            print("-" * 50)