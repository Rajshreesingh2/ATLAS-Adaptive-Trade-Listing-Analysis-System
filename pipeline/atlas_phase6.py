"""
ATLAS — Phase 6: GenAI RAG Shopping Assistant
LangChain + Google Gemini + ChromaDB
Answers any product question using 300,000 product records
"""

import os
import json
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(BASE)
PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS    = os.path.join(ROOT, "models", "genai")
os.makedirs(MODELS, exist_ok=True)

GEMINI_API_KEY = "AIzaSyDJLctpBOF0jA9OYPW-ZCIuUI8RXNXnrI0"

# ── Install check ─────────────────────────────────────────────
def check_installs():
    missing = []
    try: import chromadb
    except: missing.append("chromadb")
    try: import langchain
    except: missing.append("langchain")
    try: import langchain_google_genai
    except: missing.append("langchain-google-genai")
    try: import langchain_community
    except: missing.append("langchain-community")
    try: import google.generativeai
    except: missing.append("google-generativeai")

    if missing:
        print(f"Installing: {', '.join(missing)}")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet"] + missing)
        print("Installation complete")

check_installs()

# ── Step 1: Load Product Data ─────────────────────────────────
def load_products(sample_size=5000):
    print(f"\n[Step 1] Loading product data...")

    path = os.path.join(PROCESSED, "atlas_nlp_dataset.csv")
    if not os.path.exists(path):
        path = os.path.join(PROCESSED, "atlas_master_dataset.csv")

    df = pd.read_csv(path, low_memory=False)
    df = df[df["product_name"].notna()].copy()

    # Filter good quality products
    if "rating" in df.columns:
        df = df[df["rating"].fillna(0) >= 3.0]

    df = df.head(sample_size).reset_index(drop=True)
    print(f"  Products loaded: {len(df):,}")
    return df


# ── Step 2: Build Product Documents ──────────────────────────
def build_documents(df):
    print(f"\n[Step 2] Building product documents...")

    documents = []
    metadatas = []
    ids       = []

    for idx, row in df.iterrows():
        name     = str(row.get("product_name", ""))[:150]
        price    = row.get("price", 0) or 0
        rating   = row.get("rating", 0) or 0
        reviews  = row.get("review_count", 0) or 0
        cat      = str(row.get("predicted_category",
                                row.get("category", "General")))
        brand    = str(row.get("brand_extracted", "Unknown"))
        sentiment= str(row.get("sentiment_label", "Neutral"))
        review   = str(row.get("clean_review",
                                row.get("review_text", "")))[:200]
        features = str(row.get("features_mentioned", ""))

        # Rich document text for embedding
        doc = f"""Product: {name}
Category: {cat}
Brand: {brand}
Price: ${price:.2f}
Rating: {rating:.1f}/5.0
Reviews: {int(reviews)} customer reviews
Sentiment: {sentiment}
Features: {features if features != 'none' else 'General product'}
Review excerpt: {review if len(review) > 5 else 'No review available'}"""

        documents.append(doc)
        metadatas.append({
            "product_name": name[:100],
            "category":     cat,
            "price":        float(price),
            "rating":       float(rating),
            "brand":        brand,
            "sentiment":    sentiment,
            "product_idx":  idx,
        })
        ids.append(f"prod_{idx}")

    print(f"  Documents built: {len(documents):,}")
    return documents, metadatas, ids


# ── Step 3: Build ChromaDB Vector Store ──────────────────────
def build_vector_store(documents, metadatas, ids):
    print(f"\n[Step 3] Building ChromaDB vector store...")

    import chromadb
    from chromadb.utils import embedding_functions

    # Use sentence transformers for embeddings (free, local)
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        print("  Using SentenceTransformer embeddings (all-MiniLM-L6-v2)")
    except Exception as e:
        print(f"  SentenceTransformer failed: {e}")
        ef = embedding_functions.DefaultEmbeddingFunction()
        print("  Using default embeddings")

    # Create persistent client
    db_path = os.path.join(MODELS, "chroma_db")
    client  = chromadb.PersistentClient(path=db_path)

    # Delete existing collection if exists
    try:
        client.delete_collection("atlas_products")
    except:
        pass

    collection = client.create_collection(
        name="atlas_products",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Add in batches
    batch_size = 500
    total      = len(documents)

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"  Indexed {end:,}/{total:,} products...")

    print(f"  Vector store built: {collection.count():,} products indexed")
    return collection, client


# ── Step 4: Build RAG Chain ───────────────────────────────────
def build_rag_chain(collection):
    print(f"\n[Step 4] Building RAG chain with Google Gemini...")

    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    def retrieve(query, n_results=5, category_filter=None):
        """Retrieve relevant products from vector store."""
        where = None
        if category_filter:
            where = {"category": {"$eq": category_filter}}

        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
        except:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )

        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            retrieved.append({
                "document":   doc,
                "metadata":   meta,
                "similarity": round(1 - dist, 3)
            })
        return retrieved

    def generate_answer(query, retrieved_docs, chat_history=None):
        """Generate answer using Gemini with retrieved context."""

        # Build context from retrieved products
        context = ""
        for i, item in enumerate(retrieved_docs, 1):
            meta = item["metadata"]
            context += f"\nProduct {i} (Similarity: {item['similarity']:.2f}):\n"
            context += item["document"] + "\n"
            context += "-" * 40 + "\n"

        # Build conversation history
        history_text = ""
        if chat_history:
            for turn in chat_history[-3:]:  # Last 3 turns
                history_text += f"User: {turn['user']}\n"
                history_text += f"Assistant: {turn['assistant']}\n\n"

        prompt = f"""You are ATLAS, an expert AI shopping assistant with access to a database of 300,000+ products.

Your role is to help users find the best products, compare options, detect fake reviews, and make smart purchasing decisions.

PRODUCT DATABASE CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {query}

INSTRUCTIONS:
- Answer based on the product context provided above
- Be specific — mention actual product names, prices, and ratings from the context
- If asking for recommendations, rank products and explain why
- Mention any concerns (low rating, possible fake reviews, high price)
- Keep response concise but informative (3-5 sentences or a short list)
- If the context doesn't have relevant products, say so honestly
- Always mention the price and rating when recommending a product

RESPONSE:"""

        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."

    return retrieve, generate_answer


# ── Step 5: Interactive Chat Loop ────────────────────────────
def run_chat(retrieve, generate_answer):
    print(f"\n{'='*60}")
    print(f"ATLAS AI Shopping Assistant")
    print(f"Powered by Google Gemini + ChromaDB RAG")
    print(f"Type 'quit' to exit | Type 'clear' to reset history")
    print(f"{'='*60}")

    chat_history = []
    example_queries = [
        "What are the best wireless headphones?",
        "I need a kitchen appliance for making smoothies",
        "Show me highly rated electronics under $50",
        "What gaming accessories do you have?",
        "Compare your best rated products",
    ]

    print(f"\nExample questions you can ask:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat...")
            break

        if not query:
            continue

        if query.lower() == "quit":
            print("Goodbye!")
            break

        if query.lower() == "clear":
            chat_history = []
            print("Chat history cleared.")
            continue

        print("ATLAS: Searching product database...")

        # Retrieve relevant products
        retrieved = retrieve(query, n_results=5)

        # Generate answer
        answer = generate_answer(query, retrieved, chat_history)

        print(f"\nATLAS: {answer}")

        # Show source products
        print(f"\n  [Sources: {len(retrieved)} products retrieved]")
        for i, item in enumerate(retrieved[:3], 1):
            meta = item["metadata"]
            print(f"  {i}. {meta['product_name'][:50]} "
                  f"| ${meta['price']:.2f} | ⭐{meta['rating']:.1f} "
                  f"| Relevance: {item['similarity']:.2f}")
        print()

        # Save to history
        chat_history.append({"user": query, "assistant": answer})


# ── Step 6: Demo Queries ──────────────────────────────────────
def run_demo(retrieve, generate_answer):
    print(f"\n[Step 6] Running demo queries...")
    print(f"{'='*60}")

    demo_queries = [
        "What are the best wireless headphones you have?",
        "I need a kitchen appliance under $50 with good reviews",
        "Show me the top rated gaming products",
        "What smart home products do you recommend?",
        "Which products have the most suspicious reviews?",
    ]

    results = []
    for query in demo_queries:
        print(f"\nQuery: {query}")
        retrieved = retrieve(query, n_results=5)
        answer    = generate_answer(query, retrieved)
        print(f"ATLAS: {answer[:300]}...")
        print(f"Sources: {len(retrieved)} products | "
              f"Top match: {retrieved[0]['metadata']['product_name'][:40] if retrieved else 'None'}")
        results.append({
            "query":       query,
            "answer":      answer,
            "n_retrieved": len(retrieved),
            "top_product": retrieved[0]["metadata"]["product_name"] if retrieved else "",
        })

    return results


# ── Step 7: Save RAG Configuration ───────────────────────────
def save_config(n_products, demo_results):
    print(f"\n[Step 7] Saving RAG configuration...")

    config = {
        "model":           "Google Gemini 1.5 Flash",
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_store":    "ChromaDB",
        "n_products":      n_products,
        "retrieval_k":     5,
        "demo_queries":    len(demo_results),
        "capabilities": [
            "Product search and recommendation",
            "Price comparison",
            "Review quality assessment",
            "Category-filtered search",
            "Multi-turn conversation",
            "Contextual follow-up questions",
        ],
        "built_at": datetime.now().isoformat(),
    }

    with open(os.path.join(MODELS, "rag_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(MODELS, "demo_results.json"), "w") as f:
        json.dump(demo_results, f, indent=2)

    print(f"  Config saved to models/genai/rag_config.json")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ATLAS — Phase 6: GenAI RAG Shopping Assistant")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load and index products
    df                          = load_products(sample_size=5000)
    documents, metadatas, ids   = build_documents(df)
    collection, client          = build_vector_store(documents, metadatas, ids)
    retrieve, generate_answer   = build_rag_chain(collection)

    # Run demo
    demo_results = run_demo(retrieve, generate_answer)

    # Save config
    save_config(len(df), demo_results)

    print(f"\n{'='*60}")
    print(f"PHASE 6 COMPLETE")
    print(f"{'='*60}")
    print(f"Products indexed:    {len(df):,}")
    print(f"Vector store:        ChromaDB (persistent)")
    print(f"LLM:                 Google Gemini 1.5 Flash")
    print(f"Embeddings:          all-MiniLM-L6-v2")
    print(f"Demo queries:        {len(demo_results)}")
    print(f"{'='*60}")

    # Start interactive chat
    print(f"\nStarting interactive chat mode...")
    print(f"(Press Ctrl+C to skip chat and exit)")
    try:
        run_chat(retrieve, generate_answer)
    except KeyboardInterrupt:
        print("\nChat skipped.")

    print(f"\n[DONE] Phase 6 complete!")
    print(f"RAG system saved to: models/genai/")
    print("=" * 60)


if __name__ == "__main__":
    main()
