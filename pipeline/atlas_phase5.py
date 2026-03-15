"""
ATLAS — Phase 5: Recommendation System (Fixed)
Hybrid recommender: Collaborative + Content-based + NLP-enhanced
Fixed: evaluation logic, category classification, query matching
"""

import os
import json
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(BASE)
PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS    = os.path.join(ROOT, "models", "recommender")
os.makedirs(MODELS, exist_ok=True)


# ── Step 1: Load & Fix Categories ────────────────────────────
def load_data(sample_size=10000):
    print(f"\n[Step 1] Loading data...")
    path = os.path.join(PROCESSED, "atlas_nlp_dataset.csv")
    if not os.path.exists(path):
        path = os.path.join(PROCESSED, "atlas_master_dataset.csv")

    df = pd.read_csv(path, low_memory=False).head(sample_size)
    df = df[df["product_name"].notna()].copy()
    df = df.reset_index(drop=True)
    df["product_idx"] = df.index

    # Fix categories using better keyword matching
    category_keywords = {
        "Electronics":   ["laptop", "computer", "desktop", "processor", "ram",
                          "ssd", "monitor", "keyboard", "mouse", "charger", "cable",
                          "usb", "hdmi", "battery", "phone", "tablet", "ipad"],
        "Audio":         ["headphone", "earphone", "speaker", "earbud", "bluetooth",
                          "microphone", "soundbar", "audio", "bass", "stereo", "amp"],
        "Camera":        ["camera", "lens", "tripod", "dslr", "mirrorless", "gopro",
                          "photo", "video", "flash", "filter", "memory card"],
        "Gaming":        ["gaming", "game", "console", "playstation", "xbox",
                          "nintendo", "controller", "joystick", "fps", "gpu"],
        "Smart Home":    ["smart", "alexa", "google home", "security camera", "ring",
                          "nest", "thermostat", "doorbell", "sensor", "automation"],
        "Wearables":     ["watch", "smartwatch", "fitness tracker", "band",
                          "garmin", "fitbit", "apple watch", "heart rate"],
        "Kitchen":       ["kitchen", "cooking", "blender", "mixer", "microwave",
                          "oven", "coffee", "kettle", "air fryer", "toaster", "pot"],
        "Fashion":       ["shirt", "dress", "pants", "shoes", "jacket", "bag",
                          "wallet", "hat", "clothing", "apparel", "wear"],
        "Books":         ["book", "novel", "textbook", "guide", "paperback",
                          "hardcover", "ebook", "reading", "literature"],
        "Sports":        ["sport", "fitness", "gym", "yoga", "running", "cycling",
                          "swimming", "tennis", "football", "basketball", "workout"],
        "Beauty":        ["beauty", "skincare", "makeup", "lipstick", "foundation",
                          "serum", "moisturizer", "shampoo", "perfume", "cosmetic"],
        "Toys":          ["toy", "lego", "puzzle", "doll", "action figure",
                          "board game", "kids", "children", "play", "educational"],
        "Automotive":    ["car", "truck", "vehicle", "auto", "tire", "brake",
                          "motor", "engine", "dashboard", "polaris", "rzr", "utv",
                          "motorcycle", "trailer", "towing", "ramp"],
        "Office":        ["office", "desk", "chair", "printer", "scanner",
                          "paper", "pen", "notebook", "stapler", "filing"],
        "Tools":         ["tool", "drill", "wrench", "screwdriver", "hammer",
                          "saw", "plier", "socket", "epoxy", "sealant", "adhesive"],
        "Home":          ["home", "furniture", "bedding", "curtain", "lamp",
                          "rug", "storage", "organizer", "frame", "decor"],
    }

    def classify(text):
        if not isinstance(text, str):
            return "Other"
        text_lower = text.lower()
        scores     = {}
        for cat, kws in category_keywords.items():
            score = sum(2 if kw in text_lower else 0
                       for kw in kws if len(kw) > 4)
            score += sum(1 if kw in text_lower else 0
                        for kw in kws if len(kw) <= 4)
            if score > 0:
                scores[cat] = score
        return max(scores, key=scores.get) if scores else "Other"

    df["category_fixed"] = df["product_name"].apply(classify)

    # Show distribution
    dist = df["category_fixed"].value_counts().head(10)
    print(f"  Products loaded: {len(df):,}")
    print(f"  Top categories (fixed):")
    for cat, count in dist.items():
        print(f"    {cat:<20} {count:>6,}")

    return df


# ── Step 2: Build Rich Content Text ──────────────────────────
def build_content_text(df):
    print(f"\n[Step 2] Building content text...")

    def make_content(row):
        parts = []
        name  = str(row.get("product_name", ""))
        if name: parts.append(name)
        if name: parts.append(name)  # repeat title for weight

        cat = str(row.get("category_fixed", ""))
        if cat and cat != "Other": parts.append(cat)

        brand = str(row.get("brand_extracted", ""))
        if brand and brand not in ["Unknown", "nan"]: parts.append(brand)

        review = str(row.get("clean_review", row.get("review_text", "")))
        if len(review) > 10: parts.append(review[:200])

        features = str(row.get("features_mentioned", ""))
        if features and features != "none": parts.append(features.replace(",", " "))

        return " ".join(parts).strip()

    df["content_text"] = df.apply(make_content, axis=1)
    avg_len = df["content_text"].apply(len).mean()
    print(f"  Average content length: {avg_len:.0f} chars")
    return df


# ── Step 3: TF-IDF Content Vectors ───────────────────────────
def build_tfidf(df, max_features=8000, n_components=150):
    print(f"\n[Step 3] Building TF-IDF vectors...")

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        stop_words="english"
    )
    tfidf_matrix = tfidf.fit_transform(df["content_text"])
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    n_comp = min(n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd    = TruncatedSVD(n_components=n_comp, random_state=42)
    vecs   = svd.fit_transform(tfidf_matrix)
    print(f"  Reduced vectors: {vecs.shape}")
    print(f"  Explained variance: {svd.explained_variance_ratio_.sum()*100:.1f}%")

    return vecs, tfidf, svd, tfidf_matrix


# ── Step 4: Simulate Better Interactions ─────────────────────
def simulate_interactions(df, n_users=500):
    print(f"\n[Step 4] Simulating user interactions...")
    np.random.seed(42)

    # Group products by category for realistic simulation
    cat_groups = df.groupby("category_fixed").apply(
        lambda x: x.index.tolist()
    ).to_dict()

    interactions = []
    for user_id in range(n_users):
        # Each user has 1-3 preferred categories
        n_pref_cats = np.random.randint(1, 4)
        pref_cats   = np.random.choice(list(cat_groups.keys()),
                                        size=min(n_pref_cats, len(cat_groups)),
                                        replace=False)

        user_products = []
        # 70% from preferred categories
        for cat in pref_cats:
            cat_products = cat_groups.get(cat, [])
            if cat_products:
                n          = np.random.randint(3, 8)
                sample     = np.random.choice(cat_products,
                                               size=min(n, len(cat_products)),
                                               replace=False)
                user_products.extend([(p, np.random.uniform(3.5, 5.0)) for p in sample])

        # 30% random exploration
        n_random = max(2, len(user_products) // 3)
        random_prods = np.random.choice(len(df), size=n_random, replace=False)
        user_products.extend([(p, np.random.uniform(1.0, 3.5)) for p in random_prods])

        # Deduplicate
        seen = set()
        for prod_idx, score in user_products:
            if prod_idx not in seen:
                seen.add(prod_idx)
                interactions.append({
                    "user_id":     user_id,
                    "product_idx": int(prod_idx),
                    "score":       round(float(score), 1),
                    "category":    df.iloc[int(prod_idx)]["category_fixed"],
                })

    idf = pd.DataFrame(interactions)
    print(f"  Total interactions: {len(idf):,}")
    print(f"  Avg interactions per user: {len(idf)/n_users:.1f}")
    return idf


# ── Step 5: Collaborative Filter ─────────────────────────────
def build_collab(interactions_df, n_products, n_components=50):
    print(f"\n[Step 5] Building collaborative filter...")

    n_users    = interactions_df["user_id"].nunique()
    user_item  = np.zeros((n_users, n_products), dtype=np.float32)

    for _, row in interactions_df.iterrows():
        uid = int(row["user_id"])
        pid = int(row["product_idx"])
        if uid < n_users and pid < n_products:
            user_item[uid, pid] = row["score"]

    # Normalise rows
    row_norms = np.linalg.norm(user_item, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    user_item_norm = user_item / row_norms

    n_comp       = min(n_components, n_users - 1, n_products - 1)
    svd          = TruncatedSVD(n_components=n_comp, random_state=42)
    user_factors = svd.fit_transform(user_item_norm)
    item_factors = svd.components_.T

    print(f"  User-item matrix: {user_item.shape} | Sparsity: {(user_item==0).mean()*100:.1f}%")
    print(f"  User factors: {user_factors.shape}")
    print(f"  Item factors: {item_factors.shape}")
    print(f"  Explained variance: {svd.explained_variance_ratio_.sum()*100:.1f}%")

    return user_factors, item_factors


# ── Step 6: Feature Scores ────────────────────────────────────
def compute_feature_scores(df):
    print(f"\n[Step 6] Computing feature scores...")
    scaler = MinMaxScaler()

    rating     = df["rating"].fillna(3.0).clip(1, 5).values
    rating_s   = scaler.fit_transform(rating.reshape(-1, 1)).flatten()

    rev_col    = "review_count" if "review_count" in df.columns else "popularity_score"
    reviews    = df[rev_col].fillna(0).values
    pop_s      = scaler.fit_transform(np.log1p(reviews).reshape(-1, 1)).flatten()

    if "value_score" in df.columns:
        val_s  = scaler.fit_transform(df["value_score"].fillna(50).values.reshape(-1,1)).flatten()
    else:
        val_s  = np.ones(len(df)) * 0.5

    if "sentiment_compound" in df.columns:
        sent_s = (df["sentiment_compound"].fillna(0).values + 1) / 2
    else:
        sent_s = np.ones(len(df)) * 0.5

    if "fake_review_score" in df.columns:
        auth_s = 1 - df["fake_review_score"].fillna(0).values / 100
    else:
        auth_s = np.ones(len(df))

    quality = (rating_s * 0.35 + pop_s * 0.20 +
               val_s   * 0.20 + sent_s * 0.15 + auth_s * 0.10)

    scores_df = pd.DataFrame({
        "rating_score":       rating_s,
        "popularity_score":   pop_s,
        "value_score":        val_s,
        "sentiment_score":    sent_s,
        "authenticity_score": auth_s,
        "quality_score":      quality,
    })
    print(f"  Avg quality score: {quality.mean():.3f}")
    return scores_df


# ── Step 7: Hybrid Recommender Class ─────────────────────────
class HybridRecommender:
    def __init__(self, df, user_factors, item_factors,
                 content_vecs, feature_scores,
                 collab_w=0.30, content_w=0.45, feature_w=0.25):
        self.df            = df.reset_index(drop=True)
        self.user_factors  = user_factors
        self.item_factors  = item_factors
        self.content_vecs  = content_vecs
        self.feature_scores= feature_scores
        self.collab_w      = collab_w
        self.content_w     = content_w
        self.feature_w     = feature_w
        self.n_items       = len(df)

    def _normalize(self, arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8: return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    def recommend_for_user(self, user_id, top_k=10, exclude_idxs=None):
        exclude = set(exclude_idxs or [])
        feat    = self.feature_scores["quality_score"].values

        if user_id < len(self.user_factors):
            collab  = self._normalize(self.user_factors[user_id] @ self.item_factors.T)
            scores  = self.collab_w * collab + self.feature_w * feat
        else:
            scores  = feat.copy()

        for idx in exclude:
            if idx < len(scores): scores[idx] = -1

        top = np.argsort(scores)[::-1][:top_k]
        return self._fmt(top, scores)

    def recommend_similar(self, product_idx, top_k=10):
        if product_idx >= len(self.content_vecs):
            return self.recommend_popular(top_k)

        q_vec  = self.content_vecs[product_idx].reshape(1, -1)
        sims   = cosine_similarity(q_vec, self.content_vecs)[0]
        sims[product_idx] = -1
        feat   = self.feature_scores["quality_score"].values
        scores = self.content_w * sims + self.feature_w * feat
        top    = np.argsort(scores)[::-1][:top_k]
        return self._fmt(top, scores)

    def recommend_by_query(self, query, tfidf, svd, top_k=10,
                            category_filter=None):
        q_tfidf = tfidf.transform([query])
        q_vec   = svd.transform(q_tfidf)
        sims    = cosine_similarity(q_vec, self.content_vecs)[0]
        feat    = self.feature_scores["quality_score"].values
        scores  = self.content_w * sims + self.feature_w * feat

        if category_filter:
            mask = self.df["category_fixed"] == category_filter
            scores[~mask.values] *= 0.3

        top = np.argsort(scores)[::-1][:top_k]
        return self._fmt(top, scores)

    def recommend_popular(self, top_k=10):
        scores = self.feature_scores["quality_score"].values
        top    = np.argsort(scores)[::-1][:top_k]
        return self._fmt(top, scores)

    def _fmt(self, indices, scores):
        results = []
        for rank, idx in enumerate(indices, 1):
            if idx >= self.n_items: continue
            row = self.df.iloc[idx]
            results.append({
                "rank":         rank,
                "product_idx":  int(idx),
                "product_name": str(row.get("product_name", ""))[:70],
                "category":     str(row.get("category_fixed", "Other")),
                "price":        float(row.get("price", 0) or 0),
                "rating":       float(row.get("rating", 0) or 0),
                "score":        round(float(scores[idx]), 4),
                "sentiment":    str(row.get("sentiment_label", "N/A")),
            })
        return results


# ── Step 8: Proper Evaluation ─────────────────────────────────
def evaluate(recommender, interactions_df, k=10):
    print(f"\n[Step 8] Evaluating Precision@{k} and Recall@{k}...")

    precisions, recalls, ndcgs = [], [], []

    for user_id in interactions_df["user_id"].unique()[:200]:
        user_df  = interactions_df[interactions_df["user_id"] == user_id]
        # Relevant = products with high score (top half of user interactions)
        threshold = user_df["score"].median()
        relevant  = set(user_df[user_df["score"] >= threshold]["product_idx"].tolist())

        if len(relevant) == 0: continue

        seen  = user_df["product_idx"].tolist()
        recs  = recommender.recommend_for_user(user_id, top_k=k,
                                                exclude_idxs=seen)
        rec_idxs = [r["product_idx"] for r in recs]
        rec_set  = set(rec_idxs)

        hits      = len(relevant & rec_set)
        precision = hits / k
        recall    = hits / len(relevant)

        # NDCG
        dcg  = sum(1 / np.log2(i + 2) for i, idx in enumerate(rec_idxs) if idx in relevant)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ndcg = dcg / idcg if idcg > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

    p = np.mean(precisions)
    r = np.mean(recalls)
    n = np.mean(ndcgs)
    f = 2 * p * r / (p + r + 1e-8)

    print(f"  Precision@{k}:  {p:.4f}")
    print(f"  Recall@{k}:     {r:.4f}")
    print(f"  NDCG@{k}:       {n:.4f}")
    print(f"  F1@{k}:         {f:.4f}")

    return {f"precision@{k}": float(p), f"recall@{k}": float(r),
            f"ndcg@{k}": float(n), f"f1@{k}": float(f)}


# ── Step 9: Demo ──────────────────────────────────────────────
def demo(recommender, tfidf, svd, df):
    print(f"\n[Step 9] Demo Recommendations")
    print(f"{'='*60}")

    # User recommendations
    print(f"\n User 42 — Top 5 personalised recommendations:")
    seen = [0, 1, 2, 3, 4]
    for r in recommender.recommend_for_user(42, top_k=5, exclude_idxs=seen):
        print(f"  {r['rank']}. [{r['category']}] {r['product_name'][:55]}")
        print(f"     Rating: {r['rating']:.1f} | Price: ${r['price']:.2f} | Score: {r['score']:.3f}")

    # Query-based
    queries = [
        ("best wireless headphones for gaming",  "Audio"),
        ("laptop for programming under 1000",    "Electronics"),
        ("smart home security camera wifi",      "Smart Home"),
        ("running shoes for marathon training",  "Sports"),
    ]
    for query, cat in queries:
        print(f"\n Query: '{query}'")
        recs = recommender.recommend_by_query(query, tfidf, svd, top_k=3,
                                               category_filter=cat)
        if not recs:
            recs = recommender.recommend_by_query(query, tfidf, svd, top_k=3)
        for r in recs:
            print(f"  {r['rank']}. [{r['category']}] {r['product_name'][:55]}")
            print(f"     Score: {r['score']:.3f}")

    # Similar products
    # Find an electronics product first
    elec_idx = df[df["category_fixed"] == "Electronics"].index[0] if \
        len(df[df["category_fixed"] == "Electronics"]) > 0 else 0
    print(f"\n Products similar to: {df.iloc[elec_idx]['product_name'][:50]}")
    for r in recommender.recommend_similar(elec_idx, top_k=3):
        print(f"  {r['rank']}. [{r['category']}] {r['product_name'][:55]}")
        print(f"     Score: {r['score']:.3f}")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ATLAS — Phase 5: Recommendation System")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    df             = load_data(sample_size=10000)
    df             = build_content_text(df)
    content_vecs, tfidf, svd, _ = build_tfidf(df)
    interactions   = simulate_interactions(df, n_users=500)
    user_f, item_f = build_collab(interactions, n_products=len(df))
    feat_scores    = compute_feature_scores(df)

    rec = HybridRecommender(df, user_f, item_f, content_vecs, feat_scores)

    metrics = evaluate(rec, interactions, k=10)
    demo(rec, tfidf, svd, df)

    # Save
    results = {
        "model":          "Hybrid Recommender v2",
        "components":     ["Collaborative SVD", "Content TF-IDF + SVD", "Feature scoring"],
        "weights":        {"collaborative": 0.30, "content": 0.45, "feature": 0.25},
        "n_products":     len(df),
        "n_users":        500,
        "n_interactions": len(interactions),
        "metrics":        metrics,
        "trained_at":     datetime.now().isoformat(),
    }
    with open(os.path.join(MODELS, "recommender_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.save(os.path.join(MODELS, "content_vectors.npy"), content_vecs)
    np.save(os.path.join(MODELS, "item_factors.npy"),    item_f)

    product_index = df[["product_idx", "product_id", "product_name",
                         "category_fixed", "price", "rating"]].copy()
    product_index.to_csv(os.path.join(MODELS, "product_index.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"PHASE 5 RESULTS")
    print(f"{'='*60}")
    print(f"Products indexed:    {len(df):,}")
    print(f"Users simulated:     500")
    print(f"Interactions:        {len(interactions):,}")
    for k, v in metrics.items():
        print(f"{k:<20} {v:.4f}")
    print(f"{'='*60}")
    print(f"\n[DONE] Phase 5 complete!")
    print(f"Models saved to: models/recommender/")
    print("=" * 60)


if __name__ == "__main__":
    main()