# evaluate.py
import os
import json
from evaluation_set import EVAL_QUESTIONS
from app import load_chat_engine
import time
from sentence_transformers import SentenceTransformer, util


def evaluate():
    # Load chat engine
    chat_engine = load_chat_engine()

    # Embedding model (same as app.py)
    model_path = os.path.abspath("./bge-m3")
    sim_model = SentenceTransformer(model_path)

    results = []
    total_score = 0.0

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        question = q["question"]
        ground_truth = q["ground_truth"]

        print(f"\nQ{i}: {question}")

        # Run one-shot query
        try:
            response = chat_engine.chat(question)
            answer = getattr(response, "response", None) or str(response)
        except Exception as e:
            answer = ""
            print(f"Error during query: {e}")

        # 1. Semantic similarity
        sem_score = 0.0
        if answer.strip():
            embeddings = sim_model.encode([answer, ground_truth], convert_to_tensor=True)
            sem_score = util.cos_sim(embeddings[0], embeddings[1]).item()

        # 2. Exact match check
        exact_match = int(answer.strip().lower() == ground_truth.strip().lower())

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "similarity_score": sem_score,
            
        })

        total_score += sem_score
        print(f"Answer: {answer}")
        print(f"Similarity Score: {sem_score:.4f}")
        time.sleep(2)

    avg_score = total_score / len(EVAL_QUESTIONS)
    print("\n--- Evaluation Summary ---")
    print(f"Average Semantic Similarity: {avg_score:.4f}")

    # Save results
    with open("rag_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    evaluate()
