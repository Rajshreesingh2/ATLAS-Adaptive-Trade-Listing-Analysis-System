"""
ATLAS — Phase 2: NLP Pipeline
Sentiment Analysis + Text Classification + NER + Fake Review Detection
Runs on the 300,000 record master dataset from Phase 1
"""

import os
import json
import pandas as pd
import numpy as np
import re
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(BASE)
PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS    = os.path.join(ROOT, "models", "nlp")
os.makedirs(MODELS, exist_ok=True)

# ── Step 1: Load Master Dataset ───────────────────────────────
def load_data(sample_size=50000):
    print(f"\n[Step 1] Loading master dataset...")
    path = os.path.join(PROCESSED, "atlas_master_dataset.csv")
    df   = pd.read_csv(path, low_memory=False)
    
    # Focus on records with text content
    text_df = df[df["product_name"].notna()].copy()
    text_df = text_df.head(sample_size)
    
    print(f"  Records loaded: {len(text_df):,}")
    return text_df


# ── Step 2: Text Preprocessing ────────────────────────────────
def preprocess_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_dataset(df):
    print(f"\n[Step 2] Preprocessing text...")
    
    df["clean_title"]  = df["product_name"].apply(preprocess_text)
    df["clean_review"] = df["review_text"].fillna("").apply(preprocess_text)
    df["combined_text"]= df["clean_title"] + " " + df["clean_review"]
    df["combined_text"]= df["combined_text"].str.strip()
    
    # Text length features
    df["text_length"]      = df["combined_text"].apply(len)
    df["word_count"]       = df["combined_text"].apply(lambda x: len(x.split()))
    df["has_review"]       = df["clean_review"].apply(lambda x: len(x) > 10)
    
    print(f"  Text preprocessed for {len(df):,} records")
    print(f"  Records with reviews: {df['has_review'].sum():,}")
    return df


# ── Step 3: Sentiment Analysis with VADER ────────────────────
def run_sentiment_analysis(df):
    print(f"\n[Step 3] Running sentiment analysis...")
    
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    
    try:
        sia = SentimentIntensityAnalyzer()
    except Exception:
        nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    batch_size = 1000
    total      = len(df)
    
    for i in range(0, total, batch_size):
        batch = df["combined_text"].iloc[i:i+batch_size]
        for text in batch:
            if len(text.strip()) < 3:
                sentiments.append({
                    "sentiment_positive": 0.0,
                    "sentiment_negative": 0.0,
                    "sentiment_neutral":  1.0,
                    "sentiment_compound": 0.0,
                    "sentiment_label":    "Neutral"
                })
            else:
                scores = sia.polarity_scores(text)
                if scores["compound"] >= 0.05:
                    label = "Positive"
                elif scores["compound"] <= -0.05:
                    label = "Negative"
                else:
                    label = "Neutral"
                sentiments.append({
                    "sentiment_positive": round(scores["pos"], 4),
                    "sentiment_negative": round(scores["neg"], 4),
                    "sentiment_neutral":  round(scores["neu"], 4),
                    "sentiment_compound": round(scores["compound"], 4),
                    "sentiment_label":    label
                })
        
        if (i // batch_size) % 10 == 0:
            print(f"  Progress: {min(i+batch_size, total):,}/{total:,}")
    
    sentiment_df = pd.DataFrame(sentiments)
    df           = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
    
    # Sentiment distribution
    dist = df["sentiment_label"].value_counts()
    print(f"  Sentiment distribution:")
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"    {label:<12} {count:>8,} ({pct:.1f}%)")
    
    return df


# ── Step 4: Product Category Classification ───────────────────
def classify_categories(df):
    print(f"\n[Step 4] Classifying product categories...")
    
    # Keyword-based classification (fast, no GPU needed)
    category_keywords = {
        "Electronics":      ["laptop", "computer", "phone", "tablet", "monitor", "keyboard",
                              "mouse", "processor", "gpu", "ram", "ssd", "hdmi", "usb"],
        "Audio":            ["headphone", "earphone", "speaker", "earbuds", "bluetooth audio",
                              "microphone", "amplifier", "soundbar", "subwoofer"],
        "Camera":           ["camera", "lens", "tripod", "dslr", "mirrorless", "gopro",
                              "memory card", "filter", "flash"],
        "Gaming":           ["gaming", "game", "console", "playstation", "xbox", "nintendo",
                              "controller", "joystick", "gpu", "graphics card"],
        "Smart Home":       ["smart", "alexa", "google home", "ring", "nest", "iot",
                              "automation", "sensor", "hub"],
        "Wearables":        ["watch", "smartwatch", "fitness tracker", "band", "garmin",
                              "fitbit", "apple watch", "samsung watch"],
        "Kitchen":          ["kitchen", "cooking", "blender", "mixer", "microwave", "oven",
                              "coffee", "kettle", "air fryer", "toaster"],
        "Fashion":          ["shirt", "dress", "pants", "shoes", "jacket", "bag", "wallet",
                              "sunglasses", "hat", "clothing"],
        "Books":            ["book", "novel", "textbook", "guide", "manual", "paperback",
                              "hardcover", "ebook", "reader"],
        "Sports":           ["sport", "fitness", "gym", "yoga", "running", "cycling",
                              "swimming", "tennis", "football", "basketball"],
        "Beauty":           ["beauty", "skincare", "makeup", "lipstick", "foundation",
                              "serum", "moisturizer", "shampoo", "perfume"],
        "Toys":             ["toy", "lego", "puzzle", "doll", "action figure", "board game",
                              "kids", "children", "play"],
        "Automotive":       ["car", "vehicle", "auto", "tire", "brake", "motor", "engine",
                              "dashboard", "seat cover"],
        "Office":           ["office", "desk", "chair", "printer", "scanner", "paper",
                              "pen", "notebook", "stapler"],
        "Drones":           ["drone", "quadcopter", "fpv", "dji", "aerial", "uav",
                              "propeller", "remote control"],
    }
    
    def classify_product(text):
        if not isinstance(text, str):
            return "Other"
        text_lower = text.lower()
        scores     = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score
        if scores:
            return max(scores, key=scores.get)
        return "Other"
    
    df["predicted_category"] = df["combined_text"].apply(classify_product)
    
    dist = df["predicted_category"].value_counts().head(10)
    print(f"  Top categories:")
    for cat, count in dist.items():
        print(f"    {cat:<20} {count:>8,}")
    
    return df


# ── Step 5: Named Entity Recognition ─────────────────────────
def extract_entities(df):
    print(f"\n[Step 5] Extracting named entities...")
    
    # Brand extraction using known brand list
    known_brands = [
        "apple", "samsung", "sony", "lg", "microsoft", "google", "amazon",
        "dell", "hp", "lenovo", "asus", "acer", "nvidia", "intel", "amd",
        "bose", "jbl", "beats", "sennheiser", "logitech", "razer", "corsair",
        "anker", "belkin", "tp-link", "netgear", "wd", "seagate", "sandisk",
        "nikon", "canon", "gopro", "dji", "garmin", "fitbit", "xiaomi",
        "oneplus", "realme", "oppo", "vivo", "motorola", "nokia", "huawei",
        "philips", "panasonic", "sharp", "toshiba", "hitachi", "bosch",
        "adidas", "nike", "puma", "under armour", "reebok",
    ]
    
    def extract_brand(text):
        if not isinstance(text, str):
            return "Unknown"
        text_lower = text.lower()
        for brand in known_brands:
            if brand in text_lower:
                return brand.title()
        return "Unknown"
    
    # Price extraction from text
    def extract_price_mention(text):
        if not isinstance(text, str):
            return None
        matches = re.findall(r'\$[\d,]+\.?\d*', text)
        if matches:
            price_str = matches[0].replace('$', '').replace(',', '')
            try:
                return float(price_str)
            except:
                return None
        return None
    
    # Feature keywords extraction
    feature_keywords = {
        "battery":      ["battery", "mah", "charging", "charge", "power"],
        "display":      ["screen", "display", "resolution", "4k", "hd", "oled", "lcd", "amoled"],
        "camera":       ["camera", "megapixel", "mp", "lens", "photo", "video"],
        "performance":  ["processor", "cpu", "gpu", "ram", "storage", "speed", "fast"],
        "connectivity": ["wifi", "bluetooth", "5g", "4g", "nfc", "usb", "hdmi"],
        "design":       ["slim", "lightweight", "portable", "compact", "waterproof"],
    }
    
    def extract_features(text):
        if not isinstance(text, str):
            return []
        text_lower = text.lower()
        found      = []
        for feature, keywords in feature_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found.append(feature)
        return ",".join(found) if found else "none"
    
    df["brand_extracted"]    = df["combined_text"].apply(extract_brand)
    df["price_mentioned"]    = df["combined_text"].apply(extract_price_mention)
    df["features_mentioned"] = df["combined_text"].apply(extract_features)
    df["feature_count"]      = df["features_mentioned"].apply(
        lambda x: len(x.split(",")) if x != "none" else 0
    )
    
    brand_dist = df["brand_extracted"].value_counts().head(10)
    print(f"  Top brands detected:")
    for brand, count in brand_dist.items():
        if brand != "Unknown":
            print(f"    {brand:<20} {count:>8,}")
    
    return df


# ── Step 6: Fake Review Detection ────────────────────────────
def detect_fake_reviews(df):
    print(f"\n[Step 6] Running fake review detection...")
    
    def fake_review_score(row):
        score = 0
        text  = str(row.get("clean_review", ""))
        
        # Signal 1: Very short review with high rating
        if len(text) < 20 and row.get("rating", 3) == 5:
            score += 30
        
        # Signal 2: Excessive punctuation
        exclamation_count = text.count("!")
        if exclamation_count > 3:
            score += 15
        
        # Signal 3: ALL CAPS words
        words      = text.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if len(caps_words) > 2:
            score += 20
        
        # Signal 4: Repetitive phrases
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score += 25
        
        # Signal 5: Generic positive words only
        generic_words = ["great", "good", "nice", "perfect", "love", "best", "excellent"]
        if len(words) > 0:
            generic_ratio = sum(1 for w in words if w in generic_words) / len(words)
            if generic_ratio > 0.4:
                score += 20
        
        # Signal 6: No specific product details mentioned
        if len(text) > 0 and row.get("feature_count", 0) == 0 and len(text) < 50:
            score += 10
        
        return min(score, 100)
    
    df["fake_review_score"] = df.apply(fake_review_score, axis=1)
    df["fake_review_flag"]  = df["fake_review_score"] >= 50
    
    flagged = df["fake_review_flag"].sum()
    pct     = flagged / len(df) * 100
    print(f"  Fake reviews flagged: {flagged:,} ({pct:.1f}%)")
    print(f"  Average fake score:   {df['fake_review_score'].mean():.1f}/100")
    
    return df


# ── Step 7: Aspect-Based Sentiment ────────────────────────────
def aspect_sentiment(df):
    print(f"\n[Step 7] Running aspect-based sentiment...")
    
    aspects = {
        "battery":     ["battery", "charge", "charging", "power", "mah", "last"],
        "camera":      ["camera", "photo", "picture", "image", "lens", "shoot"],
        "display":     ["screen", "display", "bright", "resolution", "color"],
        "performance": ["fast", "speed", "slow", "lag", "smooth", "performance"],
        "price":       ["price", "cost", "value", "cheap", "expensive", "worth"],
        "build":       ["build", "quality", "material", "durable", "plastic", "metal"],
        "delivery":    ["delivery", "shipping", "arrived", "package", "box"],
    }
    
    positive_words = {"good", "great", "excellent", "amazing", "best", "love",
                      "perfect", "fantastic", "awesome", "superb", "brilliant"}
    negative_words = {"bad", "poor", "terrible", "worst", "hate", "broken",
                      "defective", "disappointed", "waste", "awful", "horrible"}
    
    def get_aspect_sentiment(text):
        if not isinstance(text, str) or len(text) < 5:
            return {}
        
        text_lower = text.lower()
        words      = set(text_lower.split())
        result     = {}
        
        for aspect, keywords in aspects.items():
            if any(kw in text_lower for kw in keywords):
                pos = len(words & positive_words)
                neg = len(words & negative_words)
                if pos > neg:
                    result[aspect] = "positive"
                elif neg > pos:
                    result[aspect] = "negative"
                else:
                    result[aspect] = "neutral"
        
        return result
    
    df["aspect_sentiments"] = df["combined_text"].apply(
        lambda x: json.dumps(get_aspect_sentiment(x))
    )
    
    # Count products mentioning each aspect
    print(f"  Aspect coverage:")
    for aspect in aspects.keys():
        count = df["combined_text"].str.contains(aspect, case=False, na=False).sum()
        print(f"    {aspect:<15} mentioned in {count:,} products")
    
    return df


# ── Step 8: NLP Score & Risk Tier ────────────────────────────
def compute_nlp_scores(df):
    print(f"\n[Step 8] Computing NLP scores...")
    
    # Overall NLP quality score
    df["nlp_quality_score"] = (
        (df["sentiment_compound"].fillna(0) + 1) * 25 +           # 0-50 points from sentiment
        (df["feature_count"].fillna(0).clip(0, 5) * 5) +          # 0-25 points from feature richness
        (df["word_count"].fillna(0).clip(0, 100) / 100 * 25)      # 0-25 points from text length
    ).round(2)
    
    # Risk tier based on fake score and sentiment
    def compute_risk(row):
        fake_score = row.get("fake_review_score", 0)
        sentiment  = row.get("sentiment_compound", 0)
        rating     = row.get("rating", 3) or 3
        
        risk = 0
        if fake_score > 60:   risk += 40
        elif fake_score > 30: risk += 20
        if sentiment < -0.2:  risk += 30
        if rating < 2.5:      risk += 30
        
        if risk >= 60:   return "High Risk"
        elif risk >= 30: return "Medium Risk"
        else:            return "Low Risk"
    
    df["product_risk_tier"] = df.apply(compute_risk, axis=1)
    
    risk_dist = df["product_risk_tier"].value_counts()
    print(f"  Risk distribution:")
    for tier, count in risk_dist.items():
        pct = count / len(df) * 100
        print(f"    {tier:<15} {count:>8,} ({pct:.1f}%)")
    
    return df


# ── Step 9: Save NLP Results ──────────────────────────────────
def save_results(df):
    print(f"\n[Step 9] Saving NLP results...")
    
    output_path = os.path.join(PROCESSED, "atlas_nlp_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    # Save NLP summary
    summary = {
        "total_records":         len(df),
        "sentiment_distribution": df["sentiment_label"].value_counts().to_dict(),
        "category_distribution":  df["predicted_category"].value_counts().head(10).to_dict(),
        "fake_reviews_flagged":   int(df["fake_review_flag"].sum()),
        "fake_review_pct":        round(df["fake_review_flag"].mean() * 100, 2),
        "top_brands":             df["brand_extracted"].value_counts().head(10).to_dict(),
        "risk_distribution":      df["product_risk_tier"].value_counts().to_dict(),
        "avg_nlp_quality":        round(df["nlp_quality_score"].mean(), 2),
        "processed_at":           datetime.now().isoformat(),
        "nlp_columns":            [c for c in df.columns if c not in
                                   ["product_id","product_name","source","processed_at"]],
    }
    
    with open(os.path.join(PROCESSED, "nlp_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"  NLP summary saved to nlp_summary.json")
    
    # Save high-risk products separately
    high_risk = df[df["product_risk_tier"] == "High Risk"].copy()
    high_risk.to_csv(os.path.join(PROCESSED, "high_risk_products.csv"), index=False)
    print(f"  High risk products: {len(high_risk):,} saved separately")
    
    return df


# ── Step 10: Evaluation Metrics ──────────────────────────────
def evaluation_report(df):
    print(f"\n[Step 10] NLP Evaluation Report")
    print(f"{'='*60}")
    print(f"Total records processed:     {len(df):,}")
    print(f"Records with reviews:        {df['has_review'].sum():,}")
    print(f"")
    print(f"SENTIMENT ANALYSIS")
    print(f"  Model:                     VADER (rule-based)")
    print(f"  Positive:                  {(df['sentiment_label']=='Positive').sum():,}")
    print(f"  Negative:                  {(df['sentiment_label']=='Negative').sum():,}")
    print(f"  Neutral:                   {(df['sentiment_label']=='Neutral').sum():,}")
    print(f"  Avg compound score:        {df['sentiment_compound'].mean():.4f}")
    print(f"")
    print(f"CATEGORY CLASSIFICATION")
    print(f"  Categories detected:       {df['predicted_category'].nunique()}")
    print(f"  Coverage:                  {(df['predicted_category']!='Other').mean()*100:.1f}%")
    print(f"")
    print(f"NAMED ENTITY RECOGNITION")
    print(f"  Brands identified:         {(df['brand_extracted']!='Unknown').sum():,}")
    print(f"  Brand coverage:            {(df['brand_extracted']!='Unknown').mean()*100:.1f}%")
    print(f"  Avg features per product:  {df['feature_count'].mean():.1f}")
    print(f"")
    print(f"FAKE REVIEW DETECTION")
    print(f"  Flagged as fake:           {df['fake_review_flag'].sum():,}")
    print(f"  Fake review rate:          {df['fake_review_flag'].mean()*100:.1f}%")
    print(f"  Avg fake score:            {df['fake_review_score'].mean():.1f}/100")
    print(f"")
    print(f"PRODUCT RISK")
    print(f"  High Risk:                 {(df['product_risk_tier']=='High Risk').sum():,}")
    print(f"  Medium Risk:               {(df['product_risk_tier']=='Medium Risk').sum():,}")
    print(f"  Low Risk:                  {(df['product_risk_tier']=='Low Risk').sum():,}")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ATLAS — Phase 2: NLP Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    df = load_data(sample_size=50000)
    df = preprocess_dataset(df)
    df = run_sentiment_analysis(df)
    df = classify_categories(df)
    df = extract_entities(df)
    df = detect_fake_reviews(df)
    df = aspect_sentiment(df)
    df = compute_nlp_scores(df)
    df = save_results(df)
    evaluation_report(df)
    
    print(f"\n[DONE] Phase 2 complete!")
    print(f"NLP dataset: {len(df):,} records with {len(df.columns)} features")
    print(f"Output: data/processed/atlas_nlp_dataset.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
