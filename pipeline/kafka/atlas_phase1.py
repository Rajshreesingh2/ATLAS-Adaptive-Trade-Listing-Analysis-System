"""
ATLAS — Phase 1: Data Pipeline & Feature Engineering
Processes Amazon Canada 2.1M products + Flipkart + live APIs
Target: 300,000 clean records ready for NLP and ML
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytrends.request import TrendReq
from newsapi import NewsApiClient
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(BASE)
RAW         = os.path.join(ROOT, "data", "raw")
PROCESSED   = os.path.join(ROOT, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)

NEWS_API_KEY = "271aae15eeaf460083131fc978a5e678"

# ── Step 1: Load Amazon Canada Dataset ───────────────────────
def load_amazon_canada(sample_size=300000):
    print(f"\n[Step 1] Loading Amazon Canada dataset...")
    path = os.path.join(RAW, "amz_ca_total_products_data_processed.csv")
    
    # Load in chunks for memory efficiency
    chunks = []
    chunk_size = 50000
    total = 0
    
    for chunk in pd.read_csv(path, chunksize=chunk_size, on_bad_lines='skip'):
        chunks.append(chunk)
        total += len(chunk)
        print(f"  Loaded {total:,} rows...")
        if total >= sample_size:
            break
    
    df = pd.concat(chunks, ignore_index=True).head(sample_size)
    print(f"  Total loaded: {len(df):,} rows")
    print(f"  Columns: {df.columns.tolist()}")
    return df


# ── Step 2: Clean & Standardise ──────────────────────────────
def clean_amazon(df):
    print(f"\n[Step 2] Cleaning Amazon Canada data...")
    
    # Standardise column names
    col_map = {
        "title":              "product_name",
        "stars":              "rating",
        "reviews":            "review_count",
        "price":              "price",
        "listPrice":          "list_price",
        "isBestSeller":       "is_bestseller",
        "boughtInLastMonth":  "bought_last_month",
        "category_id":        "category",
        "imgUrl":             "image_url",
        "productURL":         "product_url",
        "asin":               "product_id",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # Add source tag
    df["source"] = "amazon_canada"
    
    # Clean price — remove $ and commas
    if "price" in df.columns:
        df["price"] = df["price"].astype(str).str.replace(r'[\$,]', '', regex=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    
    if "list_price" in df.columns:
        df["list_price"] = df["list_price"].astype(str).str.replace(r'[\$,]', '', regex=True)
        df["list_price"] = pd.to_numeric(df["list_price"], errors="coerce")
    
    # Clean rating
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    
    # Clean review count
    if "review_count" in df.columns:
        df["review_count"] = df["review_count"].astype(str).str.replace(r'[,]', '', regex=True)
        df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
    
    # Calculate discount percentage
    if "price" in df.columns and "list_price" in df.columns:
        df["discount_pct"] = ((df["list_price"] - df["price"]) / df["list_price"] * 100).clip(0, 100).round(1)
    
    # Drop rows with no product name
    if "product_name" in df.columns:
        df = df.dropna(subset=["product_name"])
        df = df[df["product_name"].str.len() > 3]
    
    # Remove duplicates
    if "product_id" in df.columns:
        df = df.drop_duplicates(subset=["product_id"])
    
    print(f"  Clean records: {len(df):,}")
    return df


# ── Step 3: Load Flipkart Data ────────────────────────────────
def load_flipkart():
    print(f"\n[Step 3] Loading Flipkart data...")
    path = os.path.join(RAW, "flipkart_drone.csv")
    
    if not os.path.exists(path):
        print("  Flipkart file not found — skipping")
        return pd.DataFrame()
    
    df = pd.read_csv(path, on_bad_lines='skip')
    df["source"] = "flipkart"
    
    # Standardise columns
    col_map = {
        "Name":          "product_name",
        "Price":         "price",
        "Actual Price":  "list_price",
        "Discount (%)":  "discount_pct",
        "Type":          "category",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # Clean price
    if "price" in df.columns:
        df["price"] = df["price"].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    
    df["product_id"] = ["FK_" + str(i) for i in range(len(df))]
    df["rating"]     = np.nan
    df["review_count"] = 0
    
    print(f"  Flipkart records: {len(df):,}")
    return df


# ── Step 4: Load Amazon Reviews ───────────────────────────────
def load_amazon_reviews():
    print(f"\n[Step 4] Loading Amazon product reviews...")
    path = os.path.join(RAW, "7817_1.csv")
    
    if not os.path.exists(path):
        print("  Reviews file not found — skipping")
        return pd.DataFrame()
    
    df = pd.read_csv(path, on_bad_lines='skip')
    df["source"] = "amazon_reviews"
    
    # Keep useful columns
    keep = ["name", "brand", "categories", "prices",
            "reviews.text", "reviews.rating", "reviews.title",
            "reviews.doRecommend", "reviews.numHelpful"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    
    col_map = {
        "name":                "product_name",
        "brand":               "brand",
        "categories":          "category",
        "reviews.text":        "review_text",
        "reviews.rating":      "rating",
        "reviews.title":       "review_title",
        "reviews.doRecommend": "recommended",
        "reviews.numHelpful":  "helpful_votes",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["product_id"] = ["REV_" + str(i) for i in range(len(df))]
    
    print(f"  Review records: {len(df):,}")
    return df


# ── Step 5: Feature Engineering ──────────────────────────────
def engineer_features(df):
    print(f"\n[Step 5] Engineering features...")
    
    # Value score — how good is the deal?
    if "rating" in df.columns and "price" in df.columns:
        df["rating_filled"]     = df["rating"].fillna(3.0)
        df["price_filled"]      = df["price"].fillna(df["price"].median())
        df["price_filled"]      = df["price_filled"].replace(0, df["price_filled"].median())
        df["value_score"]       = (df["rating_filled"] / 5.0) / (np.log1p(df["price_filled"]) + 1)
        df["value_score"]       = (df["value_score"] * 100).round(2)
    
    # Popularity score
    if "review_count" in df.columns:
        df["review_count_filled"] = df["review_count"].fillna(0)
        max_reviews               = df["review_count_filled"].quantile(0.95)
        df["popularity_score"]    = (df["review_count_filled"] / (max_reviews + 1) * 100).clip(0, 100).round(2)
    
    # Discount tier
    if "discount_pct" in df.columns:
        df["discount_tier"] = pd.cut(
            df["discount_pct"].fillna(0),
            bins=[-1, 0, 10, 25, 50, 100],
            labels=["No Discount", "Low", "Medium", "High", "Very High"]
        )
    
    # Rating tier
    if "rating" in df.columns:
        df["rating_tier"] = pd.cut(
            df["rating"].fillna(3.0),
            bins=[0, 2, 3, 4, 4.5, 5],
            labels=["Poor", "Below Average", "Average", "Good", "Excellent"]
        )
    
    # Bestseller flag
    if "is_bestseller" in df.columns:
        df["is_bestseller"] = df["is_bestseller"].fillna(False).astype(bool)
    else:
        df["is_bestseller"] = False
    
    # Text length feature
    if "product_name" in df.columns:
        df["title_length"]      = df["product_name"].fillna("").apply(len)
        df["title_word_count"]  = df["product_name"].fillna("").apply(lambda x: len(x.split()))
    
    # Price per rating point
    if "price" in df.columns and "rating" in df.columns:
        df["price_per_rating"] = (df["price"].fillna(0) / df["rating"].fillna(3.0)).round(2)
    
    print(f"  Features engineered: {len(df.columns)} total columns")
    return df


# ── Step 6: Fetch Live News Data ──────────────────────────────
def fetch_live_news(sample_size=500):
    print(f"\n[Step 6] Fetching live news from NewsAPI...")
    try:
        newsapi  = NewsApiClient(api_key=NEWS_API_KEY)
        articles = []
        
        queries = ["ecommerce products", "amazon products", "online shopping", "product reviews"]
        
        for query in queries:
            try:
                result = newsapi.get_everything(
                    q=query, language="en",
                    sort_by="publishedAt", page_size=25
                )
                for a in result.get("articles", []):
                    articles.append({
                        "product_id":   "NEWS_" + str(len(articles)),
                        "product_name": a.get("title", ""),
                        "review_text":  a.get("description", "") or a.get("content", ""),
                        "source":       "newsapi",
                        "published_at": a.get("publishedAt", ""),
                        "news_source":  a.get("source", {}).get("name", ""),
                        "url":          a.get("url", ""),
                        "rating":       np.nan,
                        "price":        np.nan,
                    })
            except Exception as e:
                print(f"  Query '{query}' failed: {e}")
                continue
        
        news_df = pd.DataFrame(articles[:sample_size])
        print(f"  News articles fetched: {len(news_df):,}")
        return news_df
    
    except Exception as e:
        print(f"  NewsAPI error: {e}")
        return pd.DataFrame()


# ── Step 7: Fetch Google Trends ───────────────────────────────
def fetch_google_trends():
    print(f"\n[Step 7] Fetching Google Trends data...")
    try:
        pytrends = TrendReq(hl="en-US", tz=330)
        
        keywords = [
            ["laptop", "smartphone", "headphones", "smartwatch"],
            ["gaming", "camera", "tablet", "earbuds"],
        ]
        
        all_trends = []
        for kw_list in keywords:
            try:
                pytrends.build_payload(kw_list, timeframe="today 3-m", geo="CA")
                interest = pytrends.interest_over_time()
                if not interest.empty:
                    interest = interest.drop(columns=["isPartial"], errors="ignore")
                    interest = interest.reset_index()
                    interest_melted = interest.melt(id_vars=["date"], var_name="keyword", value_name="search_interest")
                    all_trends.append(interest_melted)
            except Exception as e:
                print(f"  Trends batch failed: {e}")
                continue
        
        if all_trends:
            trends_df = pd.concat(all_trends, ignore_index=True)
            trends_df.to_csv(os.path.join(PROCESSED, "google_trends.csv"), index=False)
            print(f"  Trend data points: {len(trends_df):,}")
            return trends_df
        else:
            print("  No trend data retrieved")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"  Google Trends error: {e}")
        return pd.DataFrame()


# ── Step 8: Combine & Save ────────────────────────────────────
def combine_and_save(amazon_df, flipkart_df, reviews_df, news_df):
    print(f"\n[Step 8] Combining all datasets...")
    
    # Standard columns across all sources
    standard_cols = [
        "product_id", "product_name", "source", "category",
        "price", "list_price", "discount_pct", "rating",
        "review_count", "is_bestseller", "image_url",
        "review_text", "value_score", "popularity_score",
        "discount_tier", "rating_tier", "title_length",
        "title_word_count", "price_per_rating"
    ]
    
    dfs = []
    for df, name in [(amazon_df, "Amazon"), (flipkart_df, "Flipkart"),
                     (reviews_df, "Reviews"), (news_df, "News")]:
        if df is not None and len(df) > 0:
            # Add missing columns
            for col in standard_cols:
                if col not in df.columns:
                    df[col] = np.nan
            dfs.append(df[standard_cols])
            print(f"  {name}: {len(df):,} records")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["processed_at"] = datetime.now().isoformat()
    
    # Final clean
    combined = combined.dropna(subset=["product_name"])
    combined = combined[combined["product_name"].str.len() > 3]
    combined = combined.drop_duplicates(subset=["product_id"])
    
    # Save
    output_path = os.path.join(PROCESSED, "atlas_master_dataset.csv")
    combined.to_csv(output_path, index=False)
    
    print(f"\n  Total combined records: {len(combined):,}")
    print(f"  Saved to: {output_path}")
    
    # Save summary stats
    summary = {
        "total_records":       len(combined),
        "sources":             combined["source"].value_counts().to_dict(),
        "avg_rating":          round(combined["rating"].mean(), 2),
        "avg_price":           round(combined["price"].mean(), 2),
        "processed_at":        datetime.now().isoformat(),
        "columns":             combined.columns.tolist(),
    }
    with open(os.path.join(PROCESSED, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Summary saved to pipeline_summary.json")
    return combined


# ── Step 9: Data Quality Report ───────────────────────────────
def quality_report(df):
    print(f"\n[Step 9] Data Quality Report")
    print(f"{'='*50}")
    print(f"Total records:         {len(df):,}")
    print(f"Total columns:         {len(df.columns)}")
    print(f"\nSource breakdown:")
    if "source" in df.columns:
        for src, count in df["source"].value_counts().items():
            print(f"  {src:<25} {count:>8,} records")
    
    print(f"\nMissing values (top columns):")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(10)
    for col, count in missing.items():
        pct = count / len(df) * 100
        print(f"  {col:<30} {count:>8,} ({pct:.1f}%)")
    
    print(f"\nNumeric summary:")
    for col in ["price", "rating", "review_count", "discount_pct", "value_score"]:
        if col in df.columns:
            stats = df[col].describe()
            print(f"  {col:<20} mean={stats['mean']:.2f}  min={stats['min']:.2f}  max={stats['max']:.2f}")
    print(f"{'='*50}")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ATLAS — Phase 1: Data Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load all data sources
    amazon_df   = load_amazon_canada(sample_size=300000)
    amazon_df   = clean_amazon(amazon_df)
    amazon_df   = engineer_features(amazon_df)
    
    flipkart_df = load_flipkart()
    if len(flipkart_df) > 0:
        flipkart_df = engineer_features(flipkart_df)
    
    reviews_df  = load_amazon_reviews()
    
    news_df     = fetch_live_news(sample_size=500)
    
    # Fetch Google Trends (background)
    fetch_google_trends()
    
    # Combine everything
    master_df   = combine_and_save(amazon_df, flipkart_df, reviews_df, news_df)
    
    # Quality report
    quality_report(master_df)
    
    print(f"\n[DONE] Phase 1 complete!")
    print(f"Master dataset: {len(master_df):,} records")
    print(f"Output: data/processed/atlas_master_dataset.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
