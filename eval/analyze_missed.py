import json

with open("evaluation_results_20260820_132002.json", "r", encoding="utf-8") as f:
    results = json.load(f)


for result in results["detailed_results"]:
    if result["hit_score"] == 0:
        print(f"Query: {result['query']}")
        print(f"Expected Text Match: {result['expected_text_match']}")
        print("Retrieved Documents:")
        for doc in result["retrieved_docs"]:
            print(f"- {doc['content'][:300]}...")  # Print the first 300 characters of each retrieved document
