import json

with open("evaluation_results_20260821_165907.json", "r", encoding="utf-8") as f:
    results = json.load(f)


for result in results["detailed_results"]:
    if result["hit_score"] == 0:
        print(f"Raw Query: {result['raw_query']}")
        print(f"Search Query: {result['search_query']}")
        print(f"Expected Text Match: {result['expected_text_match']}")
        print("Retrieved Documents:")
        for doc in result["retrieved_docs"]:
            print(f"- {doc['content'][:300]}...")  # Print the first 300 characters of each retrieved document
