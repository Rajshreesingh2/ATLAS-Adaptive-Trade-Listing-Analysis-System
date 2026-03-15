"""
ATLAS — Phase 4: Demand Forecasting
LSTM + GRU + Transformer for 30-day demand prediction
Runs on CPU — no GPU required
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
MODELS    = os.path.join(ROOT, "models", "forecasting")
os.makedirs(MODELS, exist_ok=True)

# ── Step 1: Load Data & Create Time Series ────────────────────
def create_demand_timeseries(sample_size=50000):
    print(f"\n[Step 1] Creating demand time series...")

    df = pd.read_csv(os.path.join(PROCESSED, "atlas_master_dataset.csv"), low_memory=False)
    df = df[df["review_count"].notna()].head(sample_size).copy()

    # Use review_count as proxy for demand
    # Simulate 90-day daily demand per category
    np.random.seed(42)
    categories = df["predicted_category"].value_counts().head(10).index.tolist() \
        if "predicted_category" in df.columns else ["Electronics", "Kitchen", "Fashion"]

    time_series = []
    dates = pd.date_range(start="2023-01-01", periods=90, freq="D")

    for category in categories:
        cat_df   = df[df.get("predicted_category", pd.Series(["Other"]*len(df))) == category] \
            if "predicted_category" in df.columns else df.head(1000)
        base_demand = max(len(cat_df) // 10, 10)

        # Add trend + seasonality + noise
        trend     = np.linspace(0, base_demand * 0.2, 90)
        seasonal  = base_demand * 0.3 * np.sin(np.linspace(0, 4*np.pi, 90))
        noise     = np.random.normal(0, base_demand * 0.1, 90)
        demand    = base_demand + trend + seasonal + noise
        demand    = np.clip(demand, 1, None).round().astype(int)

        for i, (date, d) in enumerate(zip(dates, demand)):
            time_series.append({
                "date":          date,
                "category":      category,
                "demand":        d,
                "day_of_week":   date.dayofweek,
                "day_of_month":  date.day,
                "month":         date.month,
                "is_weekend":    int(date.dayofweek >= 5),
                "trend":         float(trend[i]),
            })

    ts_df = pd.DataFrame(time_series)
    ts_df.to_csv(os.path.join(PROCESSED, "demand_timeseries.csv"), index=False)
    print(f"  Categories: {len(categories)}")
    print(f"  Total records: {len(ts_df):,}")
    print(f"  Date range: {ts_df['date'].min()} to {ts_df['date'].max()}")
    return ts_df


# ── Step 2: Prepare Sequences ─────────────────────────────────
def prepare_sequences(ts_df, seq_len=30, forecast_horizon=7):
    print(f"\n[Step 2] Preparing sequences (seq_len={seq_len}, horizon={forecast_horizon})...")

    from sklearn.preprocessing import MinMaxScaler

    all_X, all_y, all_meta = [], [], []
    scalers = {}

    for category in ts_df["category"].unique():
        cat_data = ts_df[ts_df["category"] == category].sort_values("date")
        demand   = cat_data["demand"].values.astype(float)

        if len(demand) < seq_len + forecast_horizon:
            continue

        # Scale demand
        scaler = MinMaxScaler()
        demand_scaled = scaler.fit_transform(demand.reshape(-1, 1)).flatten()
        scalers[category] = scaler

        # Additional features
        features = np.column_stack([
            demand_scaled,
            cat_data["day_of_week"].values / 6.0,
            cat_data["is_weekend"].values,
            cat_data["month"].values / 12.0,
        ])

        # Create sliding window sequences
        for i in range(len(features) - seq_len - forecast_horizon + 1):
            X = features[i:i+seq_len]
            y = demand_scaled[i+seq_len:i+seq_len+forecast_horizon]
            all_X.append(X)
            all_y.append(y)
            all_meta.append(category)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)

    print(f"  Sequences created: {len(X):,}")
    print(f"  Input shape:  {X.shape}  (samples, seq_len, features)")
    print(f"  Output shape: {y.shape}  (samples, forecast_horizon)")

    return X, y, scalers


# ── Step 3: Build Models ──────────────────────────────────────
def build_lstm(input_size, hidden_size=64, num_layers=2, output_size=7, dropout=0.2):
    import torch.nn as nn

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            out     = self.dropout(out[:, -1, :])
            return self.fc(out)

    return LSTMForecaster()


def build_gru(input_size, hidden_size=64, num_layers=2, output_size=7, dropout=0.2):
    import torch.nn as nn

    class GRUForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            )

        def forward(self, x):
            out, _ = self.gru(x)
            out     = self.dropout(out[:, -1, :])
            return self.fc(out)

    return GRUForecaster()


# ── Step 4: Train Model ───────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val,
                epochs=30, batch_size=32, lr=1e-3, model_name="model"):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    device    = torch.device("cpu")
    model     = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.FloatTensor(y_train)
    X_v  = torch.FloatTensor(X_val)
    y_v  = torch.FloatTensor(y_val)

    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state    = None
    history       = {"train_loss": [], "val_loss": []}

    print(f"\n  Training {model_name}...")
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} Status")
    print(f"  {'-'*50}")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred  = model(xb)
            loss  = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            status        = "<-- best"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:<8} {train_loss:<14.6f} {val_loss:<14.6f} {status}")

    # Restore best weights
    model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.6f}")
    return model, history


# ── Step 5: Evaluate ──────────────────────────────────────────
def evaluate_model(model, X_test, y_test, scalers, model_name="model"):
    import torch

    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).numpy()

    # Metrics
    mae  = np.mean(np.abs(preds - y_test))
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    mape = np.mean(np.abs((preds - y_test) / (y_test + 1e-8))) * 100

    # Directional accuracy — did we predict up/down correctly?
    actual_dir = np.sign(y_test[:, -1] - y_test[:, 0])
    pred_dir   = np.sign(preds[:, -1]  - preds[:, 0])
    dir_acc    = np.mean(actual_dir == pred_dir)

    print(f"\n  {model_name} Evaluation:")
    print(f"    MAE:                  {mae:.4f}")
    print(f"    RMSE:                 {rmse:.4f}")
    print(f"    MAPE:                 {mape:.2f}%")
    print(f"    Directional Accuracy: {dir_acc*100:.2f}%")

    return {"mae": float(mae), "rmse": float(rmse),
            "mape": float(mape), "dir_acc": float(dir_acc)}


# ── Step 6: Baseline Comparison ───────────────────────────────
def naive_baseline(X_test, y_test):
    """Naive baseline: predict last value as constant forecast"""
    last_val = X_test[:, -1, 0:1]
    preds    = np.repeat(last_val, y_test.shape[1], axis=1)
    mae      = np.mean(np.abs(preds - y_test))
    rmse     = np.sqrt(np.mean((preds - y_test) ** 2))
    mape     = np.mean(np.abs((preds - y_test) / (y_test + 1e-8))) * 100
    print(f"\n  Naive Baseline (last value):")
    print(f"    MAE:  {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAPE: {mape:.2f}%")
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ATLAS — Phase 4: Demand Forecasting")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    import torch
    from sklearn.model_selection import train_test_split

    # Create or load time series
    ts_path = os.path.join(PROCESSED, "demand_timeseries.csv")
    if os.path.exists(ts_path):
        ts_df = pd.read_csv(ts_path, parse_dates=["date"])
        print(f"\n[Step 1] Loaded existing time series: {len(ts_df):,} records")
    else:
        ts_df = create_demand_timeseries()

    # Prepare sequences
    SEQ_LEN  = 30
    HORIZON  = 7
    X, y, scalers = prepare_sequences(ts_df, seq_len=SEQ_LEN, forecast_horizon=HORIZON)

    # Train/val/test split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)
    print(f"\n[Step 3] Split: Train={len(X_tr):,} | Val={len(X_val):,} | Test={len(X_te):,}")

    input_size = X.shape[2]
    results    = {}

    # ── Train LSTM ──
    print(f"\n[Step 4a] Training LSTM...")
    lstm_model = build_lstm(input_size=input_size, hidden_size=64,
                             num_layers=2, output_size=HORIZON)
    lstm_model, lstm_history = train_model(
        lstm_model, X_tr, y_tr, X_val, y_val,
        epochs=30, batch_size=32, model_name="LSTM"
    )
    torch.save(lstm_model.state_dict(), os.path.join(MODELS, "lstm_model.pth"))
    results["LSTM"] = evaluate_model(lstm_model, X_te, y_te, scalers, "LSTM")

    # ── Train GRU ──
    print(f"\n[Step 4b] Training GRU...")
    gru_model = build_gru(input_size=input_size, hidden_size=64,
                           num_layers=2, output_size=HORIZON)
    gru_model, gru_history = train_model(
        gru_model, X_tr, y_tr, X_val, y_val,
        epochs=30, batch_size=32, model_name="GRU"
    )
    torch.save(gru_model.state_dict(), os.path.join(MODELS, "gru_model.pth"))
    results["GRU"] = evaluate_model(gru_model, X_te, y_te, scalers, "GRU")

    # ── Naive Baseline ──
    print(f"\n[Step 5] Baseline comparison...")
    results["Naive_Baseline"] = naive_baseline(X_te, y_te)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON — 7-Day Demand Forecasting")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'Dir Acc'}")
    print(f"{'-'*60}")
    for name, metrics in results.items():
        dir_acc = f"{metrics.get('dir_acc',0)*100:.1f}%" if 'dir_acc' in metrics else "N/A"
        print(f"{name:<20} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} "
              f"{metrics['mape']:<10.2f} {dir_acc}")

    # Best model
    ml_models = {k: v for k, v in results.items() if k != "Naive_Baseline"}
    best_name = min(ml_models, key=lambda k: ml_models[k]["mae"])
    print(f"\nBest model: {best_name} (MAE: {ml_models[best_name]['mae']:.4f})")
    improvement = (results["Naive_Baseline"]["mae"] - ml_models[best_name]["mae"]) \
                   / results["Naive_Baseline"]["mae"] * 100
    print(f"Improvement over naive baseline: {improvement:.1f}%")

    # Save results
    final_results = {
        "task":             "7-day demand forecasting",
        "seq_len":          SEQ_LEN,
        "forecast_horizon": HORIZON,
        "input_features":   input_size,
        "train_samples":    len(X_tr),
        "test_samples":     len(X_te),
        "models":           results,
        "best_model":       best_name,
        "baseline_improvement_pct": float(improvement),
        "trained_at":       datetime.now().isoformat(),
    }
    with open(os.path.join(MODELS, "forecasting_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n[DONE] Phase 4 complete!")
    print(f"Models saved to: models/forecasting/")
    print(f"Results: models/forecasting/forecasting_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
