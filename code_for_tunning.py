#!/usr/bin/env python3
import os
import random
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


MODE = "train"
MODEL_TYPE = "gru"

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


seed_everything(SEED)


LAKE_NAMES = [
    "inarijärvi", "pielinen", "pyhäjärvi_säkylä", "kitkajärvi",
    "oulujärvi", "simojärvi", "pallasjärvi", "höytiäinen", "jerisjärvi",
    "juojärvi", "keitele", "kemijärvi", "kiantajärvi", "kilpisjärvi",
    "kivijärvi", "koitere", "kuortaneenjärvi", "lappajärvi", "lentua",
    "lestijärvi", "lohjanjärvi", "lokan_tekojärvi", "meikojärvi",
    "muuratjärvi", "ontojärvi", "päijänne", "raanujärvi", "ähtärinjärvi",
    "unari", "vesijärvi"
]

WL_DIR = Path("WL")
METEO_DIR = Path("METEO")

DATE_COLUMN = "date"
TARGET_COLUMN = "wl"

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20

CONTEXT_LENGTH = 60
PREDICTION_LENGTH = 15

BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 8
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REFIT_ON_TRAIN_PLUS_VAL = True
FINAL_FIXED_EPOCHS = True

OUT_DIR = Path(f"{MODEL_TYPE}_selected_model")
MODEL_PATH = OUT_DIR / "model.pt"
ARTIFACTS_PATH = OUT_DIR / "artifacts.pkl"
TUNING_RESULTS_PATH = OUT_DIR / "tuning_results.csv"
TRAINVAL_CURVE_PATH = OUT_DIR / "train_val_curve.csv"

HIDDEN_SIZES = [32, 64, 128, 256]
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
LRS = [1e-4, 3e-4, 1e-3, 3e-3]
WDS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
OPTIMIZERS = ["adam", "adamw", "rmsprop"]

MAX_EPOCHS_TUNE = 20
PATIENCE_TUNE = 4

N_RANDOM_TRIALS = 200


def split_indices(n: int) -> Tuple[int, int]:
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_end = max(0, min(train_end, n))
    val_end = max(train_end, min(val_end, n))
    return train_end, val_end


def load_lake_merged(lake: str) -> pd.DataFrame:
    wl_path = WL_DIR / f"{lake}_wl.csv"
    meteo_path = METEO_DIR / f"{lake}_meteos.csv"
    if not wl_path.exists():
        raise FileNotFoundError(f"Missing WL file: {wl_path}")
    if not meteo_path.exists():
        raise FileNotFoundError(f"Missing METEO file: {meteo_path}")
    df_wl = pd.read_csv(wl_path, parse_dates=[DATE_COLUMN])
    df_wl.columns = [DATE_COLUMN, TARGET_COLUMN]
    df_meteo = pd.read_csv(meteo_path, parse_dates=[DATE_COLUMN])
    df = pd.merge(df_wl, df_meteo, on=DATE_COLUMN, how="inner")
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    df["lake"] = lake
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in (DATE_COLUMN, "lake")]
    cols = [TARGET_COLUMN] + [c for c in cols if c != TARGET_COLUMN]
    return cols


def build_windows_for_segment(
    df: pd.DataFrame,
    seg_start: int,
    seg_end: int,
    context_len: int,
    pred_len: int,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for t in range(seg_start, seg_end):
        ctx_start = t - context_len
        tgt_end = t + pred_len
        if ctx_start < 0:
            continue
        if tgt_end > seg_end:
            continue
        ctx = df.iloc[ctx_start:t][feature_cols].to_numpy(dtype=np.float32)
        tgt = df.iloc[t:tgt_end][TARGET_COLUMN].to_numpy(dtype=np.float32)
        if ctx.shape[0] != context_len or tgt.shape[0] != pred_len:
            continue
        X_list.append(ctx)
        y_list.append(tgt)
    if not X_list:
        return (
            np.empty((0, context_len, len(feature_cols)), dtype=np.float32),
            np.empty((0, pred_len), dtype=np.float32),
        )
    return np.stack(X_list, axis=0), np.stack(y_list, axis=0)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class RNNForecaster(nn.Module):
    def __init__(self, model_type: str, n_features: int, hidden_size: int, dropout: float, pred_len: int):
        super().__init__()
        mt = model_type.lower()
        self.model_type = mt
        if mt == "lstm":
            self.rnn = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=dropout,
                batch_first=True,
                bidirectional=False,
            )
            out_dim = hidden_size
        elif mt == "gru":
            self.rnn = nn.GRU(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=dropout,
                batch_first=True,
                bidirectional=False,
            )
            out_dim = hidden_size
        else:
            raise ValueError("model_type must be one of: 'lstm', 'gru'")
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.head(last)


def make_optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, alpha=0.99)
    raise ValueError(f"Unknown optimizer: {name}")


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    mse = nn.MSELoss(reduction="sum")
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        pred = model(Xb)
        loss = mse(pred, yb)
        total += float(loss.item())
        n += yb.numel()
    return total / max(1, n)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: str,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    patience: int,
    grad_clip: float,
    record_curve: bool = False,
    early_stopping: bool = True,
) -> Tuple[nn.Module, float, List[Dict[str, float]]]:
    opt = make_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    bad = 0
    curve: List[Dict[str, float]] = []
    model.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(Xb)
            loss = mse(pred, yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += float(loss.item())

        train_mse_epoch = running / max(1, len(train_loader))
        if val_loader is not None:
            val = eval_loss(model, val_loader, device=device)
        else:
            val = float("nan")

        if record_curve:
            row = {"epoch": float(epoch), "train_mse": float(train_mse_epoch)}
            if val_loader is not None:
                row["val_mse"] = float(val)
            curve.append(row)

        if early_stopping and (val_loader is not None):
            if val < best_val - 1e-8:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if early_stopping and (best_state is not None):
        model.load_state_dict(best_state)

    if val_loader is None:
        return model, float("nan"), curve
    return model, float(best_val if early_stopping else val), curve


def build_pooled_data(
    dfs_by_lake: Dict[str, pd.DataFrame],
) -> Tuple[
    List[str],
    List[str],
    StandardScaler,
    Dict[str, Tuple[float, float]],
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    use_lakes = [l for l in LAKE_NAMES if l in dfs_by_lake]
    if not use_lakes:
        raise RuntimeError("No lakes available for pooled training.")

    feat_cols = feature_columns(dfs_by_lake[use_lakes[0]])
    meteo_cols = [c for c in feat_cols if c != TARGET_COLUMN]

    wl_scalers: Dict[str, Tuple[float, float]] = {}
    for lake in use_lakes:
        df = dfs_by_lake[lake]
        n = len(df)
        train_end, _ = split_indices(n)
        wl_train = df.iloc[:train_end][TARGET_COLUMN].to_numpy(dtype=np.float32)
        mu = float(np.mean(wl_train))
        sd = float(np.std(wl_train) + 1e-8)
        wl_scalers[lake] = (mu, sd)

    scaler_meteo = StandardScaler()
    X_rows = []
    for lake in use_lakes:
        df = dfs_by_lake[lake]
        n = len(df)
        train_end, _ = split_indices(n)
        if train_end > 0:
            X_rows.append(df.iloc[:train_end][meteo_cols].to_numpy(dtype=np.float32))
    if not X_rows:
        raise RuntimeError("No pooled TRAIN rows for meteo scaler.")
    scaler_meteo.fit(np.vstack(X_rows))

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_trainval_list, y_trainval_list = [], []

    for lake in use_lakes:
        df0 = dfs_by_lake[lake]
        df = df0.copy()
        n = len(df)
        train_end, val_end = split_indices(n)

        mu, sd = wl_scalers[lake]
        df[TARGET_COLUMN] = (df[TARGET_COLUMN].to_numpy(dtype=np.float32) - mu) / sd
        df[meteo_cols] = scaler_meteo.transform(df[meteo_cols].to_numpy(dtype=np.float32))

        Xtr, ytr = build_windows_for_segment(
            df, seg_start=CONTEXT_LENGTH, seg_end=train_end,
            context_len=CONTEXT_LENGTH, pred_len=PREDICTION_LENGTH, feature_cols=feat_cols
        )
        Xva, yva = build_windows_for_segment(
            df, seg_start=train_end, seg_end=val_end,
            context_len=CONTEXT_LENGTH, pred_len=PREDICTION_LENGTH, feature_cols=feat_cols
        )
        Xtv, ytv = build_windows_for_segment(
            df, seg_start=CONTEXT_LENGTH, seg_end=val_end,
            context_len=CONTEXT_LENGTH, pred_len=PREDICTION_LENGTH, feature_cols=feat_cols
        )

        if Xtr.shape[0] > 0:
            X_train_list.append(Xtr)
            y_train_list.append(ytr)
        if Xva.shape[0] > 0:
            X_val_list.append(Xva)
            y_val_list.append(yva)
        if Xtv.shape[0] > 0:
            X_trainval_list.append(Xtv)
            y_trainval_list.append(ytv)

    if not X_train_list or not X_val_list:
        raise RuntimeError("Insufficient pooled train/val windows across lakes.")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_trainval = np.concatenate(X_trainval_list, axis=0)
    y_trainval = np.concatenate(y_trainval_list, axis=0)

    return (
        feat_cols,
        meteo_cols,
        scaler_meteo,
        wl_scalers,
        X_train, y_train,
        X_val, y_val,
        X_trainval, y_trainval,
    )


def tune_hyperparams_random(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_features: int,
    n_trials: int = N_RANDOM_TRIALS,
    tuning_results_path: Optional[Path] = None,
) -> Dict[str, Any]:
    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    rng = np.random.default_rng(SEED)

    best_cfg: Optional[Dict[str, Any]] = None
    best_val = float("inf")
    rows = []

    for _ in range(1, n_trials + 1):
        cfg = {
            "model_type": MODEL_TYPE.lower(),
            "hidden_size": int(rng.choice(HIDDEN_SIZES)),
            "dropout": float(rng.choice(DROPOUTS)),
            "lr": float(rng.choice(LRS)),
            "weight_decay": float(rng.choice(WDS)),
            "optimizer": str(rng.choice(OPTIMIZERS)),
        }

        seed_everything(SEED)

        model = RNNForecaster(
            model_type=cfg["model_type"],
            n_features=n_features,
            hidden_size=cfg["hidden_size"],
            dropout=cfg["dropout"],
            pred_len=PREDICTION_LENGTH,
        )

        model, val_mse, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            max_epochs=MAX_EPOCHS_TUNE,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            optimizer_name=cfg["optimizer"],
            patience=PATIENCE_TUNE,
            grad_clip=GRAD_CLIP,
            record_curve=False,
            early_stopping=True,
        )

        rows.append({**cfg, "best_val_mse": float(val_mse)})

        if val_mse < best_val:
            best_val = val_mse
            best_cfg = cfg

    assert best_cfg is not None
    if tuning_results_path is not None:
        pd.DataFrame(rows).to_csv(tuning_results_path, index=False)
    return best_cfg


def save_artifacts(
    feat_cols: List[str],
    meteo_cols: List[str],
    scaler_meteo: StandardScaler,
    wl_scalers: Dict[str, Tuple[float, float]],
    best_cfg: Dict[str, Any],
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "feat_cols": feat_cols,
        "meteo_cols": meteo_cols,
        "scaler_meteo": scaler_meteo,
        "wl_scalers": wl_scalers,
        "best_cfg": best_cfg,
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "seed": SEED,
        "model_type": best_cfg.get("model_type", MODEL_TYPE.lower()),
    }
    with open(ARTIFACTS_PATH, "wb") as f:
        pickle.dump(payload, f)


def save_model(model: nn.Module):
    torch.save(model.state_dict(), MODEL_PATH)


def run_train_only(dfs_by_lake: Dict[str, pd.DataFrame]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (
        feat_cols,
        meteo_cols,
        scaler_meteo,
        wl_scalers,
        X_train, y_train,
        X_val, y_val,
        X_trainval, y_trainval,
    ) = build_pooled_data(dfs_by_lake=dfs_by_lake)

    n_features = X_train.shape[-1]

    best_cfg = tune_hyperparams_random(
        X_train, y_train, X_val, y_val,
        n_features=n_features,
        n_trials=N_RANDOM_TRIALS,
        tuning_results_path=TUNING_RESULTS_PATH,
    )

    seed_everything(SEED)
    model = RNNForecaster(
        model_type=best_cfg["model_type"],
        n_features=n_features,
        hidden_size=best_cfg["hidden_size"],
        dropout=best_cfg["dropout"],
        pred_len=PREDICTION_LENGTH,
    )

    if REFIT_ON_TRAIN_PLUS_VAL and FINAL_FIXED_EPOCHS:
        train_loader = DataLoader(WindowDataset(X_trainval, y_trainval), batch_size=BATCH_SIZE, shuffle=True)
        model, best_val_mse, curve = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            device=DEVICE,
            max_epochs=MAX_EPOCHS,
            lr=best_cfg["lr"],
            weight_decay=best_cfg["weight_decay"],
            optimizer_name=best_cfg["optimizer"],
            patience=PATIENCE,
            grad_clip=GRAD_CLIP,
            record_curve=True,
            early_stopping=False,
        )
    else:
        train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
        model, best_val_mse, curve = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            max_epochs=MAX_EPOCHS,
            lr=best_cfg["lr"],
            weight_decay=best_cfg["weight_decay"],
            optimizer_name=best_cfg["optimizer"],
            patience=PATIENCE,
            grad_clip=GRAD_CLIP,
            record_curve=True,
            early_stopping=True,
        )

    if curve:
        pd.DataFrame(curve).to_csv(TRAINVAL_CURVE_PATH, index=False)

    save_model(model)
    save_artifacts(feat_cols, meteo_cols, scaler_meteo, wl_scalers, best_cfg)


def main():
    dfs_by_lake: Dict[str, pd.DataFrame] = {lake: load_lake_merged(lake) for lake in LAKE_NAMES}
    if MODE == "train":
        run_train_only(dfs_by_lake)
        return
    raise ValueError("MODE must be 'train'")


if __name__ == "__main__":
    main()

