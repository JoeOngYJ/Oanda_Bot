from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from oanda_bot.features.feature_builder import FeatureBuilder
from oanda_bot.backtesting.costs.cost_model import CostModel
from oanda_bot.backtesting.labels.forward_return_labeler import make_labels as make_forward_labels
from oanda_bot.features.breakout_model import BreakoutModel, make_breakout_labels, train_breakout_model
from oanda_bot.features.direction_model import DirectionModel, train_direction_model
from oanda_bot.features.mean_reversion_model import MeanReversionModel, make_mean_reversion_labels, train_mean_reversion_model
from oanda_bot.features.opportunity_model import OpportunityModel, train_opportunity_model
from oanda_bot.features.regime_mlp import RegimeMLP, make_regime_targets, predict_regime_proba, train_regime_mlp
from oanda_bot.features.risk_model import RiskModel, make_risk_labels, train_risk_model

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


@dataclass
class WalkForwardConfig:
    train_months: int = 24
    val_months: int = 3
    test_months: int = 3
    step_months: int = 1
    horizon_bars: int = 8
    no_trade_band: float = 0.0
    feature_schema_version: str = "v1"
    cost_model_version: str = "v1"
    output_dir: str = "models/walkforward"
    instrument: str = "EUR_USD"


def _month_floor(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz=ts.tz)


def _hash_config(cfg: Dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _load_ohlcv(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        p = Path(path_or_df)
        if p.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df = df.sort_index()
    return df


def _rolling_windows(
    idx: pd.DatetimeIndex,
    train_months: int,
    val_months: int,
    test_months: int,
    step_months: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if idx.empty:
        return []
    start0 = _month_floor(idx.min())
    last = idx.max()
    windows = []
    cur = start0
    while True:
        tr_s = cur
        tr_e = tr_s + pd.DateOffset(months=train_months)
        va_s = tr_e
        va_e = va_s + pd.DateOffset(months=val_months)
        te_s = va_e
        te_e = te_s + pd.DateOffset(months=test_months)
        if te_s > last:
            break
        windows.append((tr_s, tr_e, va_s, va_e, te_s, te_e))
        cur = cur + pd.DateOffset(months=step_months)
    return windows


def _default_eval_fn(prob_df: pd.DataFrame) -> Dict[str, Any]:
    # Backtester-compatible summary placeholder based on probability signals.
    p = prob_df.copy()
    if p.empty:
        return {"rows": 0, "trade_rate": 0.0, "long_rate": 0.0, "avg_conf": 0.0}
    trade = (p["p_trade"] >= 0.5).astype(float)
    side = np.sign(p["p_long"] - p["p_short"])
    return {
        "rows": int(len(p)),
        "trade_rate": float(trade.mean()),
        "long_rate": float((side > 0).mean()),
        "avg_conf": float(np.maximum(p["p_long"], p["p_short"]).mean()),
    }


def _build_samples(
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    fb: FeatureBuilder,
    instrument: str,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, pd.DataFrame]:
    m15 = m15.sort_index()
    times: List[pd.Timestamp] = []
    seqs: List[np.ndarray] = []
    ctxs: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []

    for ts in m15.index:
        sl = m15.loc[:ts]
        if len(sl) < max(300, fb.seq_len):
            continue
        try:
            seq, ctx, meta = fb.build(sl, h1.loc[:ts], h4.loc[:ts], instrument=instrument)
        except Exception:
            continue
        times.append(ts)
        seqs.append(seq)
        ctxs.append(ctx)
        rows.append(
            {
                "datetime": ts,
                "close": float(meta.get("close", np.nan)),
                "atr": float(meta.get("atr", np.nan)),
                "ctx_h1_slope": float(ctx[0]) if len(ctx) > 0 else np.nan,
                "ctx_h4_slope": float(ctx[1]) if len(ctx) > 1 else np.nan,
                "ctx_h1_adx": float(ctx[2]) if len(ctx) > 2 else np.nan,
                "ctx_h4_adx": float(ctx[3]) if len(ctx) > 3 else np.nan,
                "ctx_h1_vol_pct": float(ctx[4]) if len(ctx) > 4 else np.nan,
                "ctx_h4_vol_pct": float(ctx[5]) if len(ctx) > 5 else np.nan,
                "ctx_session_asia": float(ctx[8]) if len(ctx) > 8 else np.nan,
                "ctx_session_london": float(ctx[9]) if len(ctx) > 9 else np.nan,
                "ctx_session_ny": float(ctx[10]) if len(ctx) > 10 else np.nan,
            }
        )
    if not times:
        return pd.DatetimeIndex([]), np.empty((0, fb.seq_len, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32), pd.DataFrame()
    t_idx = pd.DatetimeIndex(times)
    seq_arr = np.asarray(seqs, dtype=np.float32)
    ctx_arr = np.asarray(ctxs, dtype=np.float32)
    feat_df = pd.DataFrame(rows).set_index("datetime")
    return t_idx, seq_arr, ctx_arr, feat_df


def walkforward_train(
    *,
    m15: str | Path | pd.DataFrame,
    h1: str | Path | pd.DataFrame,
    h4: str | Path | pd.DataFrame,
    config: Optional[WalkForwardConfig] = None,
    feature_builder: Optional[FeatureBuilder] = None,
    cost_model: Optional[CostModel] = None,
    eval_fn: Optional[Callable[[pd.DataFrame], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Walk-forward training orchestration.

    - rolling windows: train 24m, val 3m, test 3m, step 1m (configurable)
    - purge/embargo by horizon H
    - train order: regime -> opportunity/direction -> specialists -> risk
    - saves weights + config hash + schema/version metadata
    - evaluates with probability-signal dataframe via eval_fn
    """
    if torch is None:
        raise ImportError("PyTorch is required for walkforward training.")

    cfg = config or WalkForwardConfig()
    fb = feature_builder or FeatureBuilder(seq_len=128)
    cm = cost_model or CostModel()
    evaluator = eval_fn or _default_eval_fn

    m15_df = _load_ohlcv(m15)
    h1_df = _load_ohlcv(h1)
    h4_df = _load_ohlcv(h4)
    m15_df = cm.add_cost_columns(m15_df, cfg.instrument)
    lbl_df = make_forward_labels(m15_df, horizon_bars=cfg.horizon_bars, no_trade_band=cfg.no_trade_band, use_costs=True)
    lab_break = make_breakout_labels(lbl_df, horizon_bars=cfg.horizon_bars)
    lab_mean = make_mean_reversion_labels(lbl_df, horizon_bars=cfg.horizon_bars)
    lab_risk = make_risk_labels(lbl_df, horizon_bars=cfg.horizon_bars, method="realized_vol")

    t_idx, seq_arr, ctx_arr, feat_df = _build_samples(lbl_df, h1_df, h4_df, fb, cfg.instrument)
    if len(t_idx) == 0:
        raise RuntimeError("No valid samples generated; check history length and data quality.")

    # Align labels to sample timestamps.
    aligned = pd.DataFrame(index=t_idx)
    for col, src in [
        ("y_opportunity", lbl_df),
        ("y_direction", lbl_df),
        ("y_breakout", lab_break),
        ("y_mean_reversion", lab_mean),
        ("y_risk", lab_risk),
    ]:
        aligned[col] = pd.to_numeric(src.reindex(t_idx)[col], errors="coerce")
    aligned = aligned.join(feat_df, how="left")
    aligned["h1_slope"] = aligned.get("ctx_h1_slope")
    aligned["h1_adx"] = aligned.get("ctx_h1_adx")
    aligned["h1_vol_pct"] = aligned.get("ctx_h1_vol_pct")
    aligned["spread_est"] = lbl_df.reindex(t_idx)["spread_est"]
    y_regime = make_regime_targets(aligned)

    # Rolling splits.
    windows = _rolling_windows(
        t_idx,
        train_months=cfg.train_months,
        val_months=cfg.val_months,
        test_months=cfg.test_months,
        step_months=cfg.step_months,
    )
    if not windows:
        raise RuntimeError("No walk-forward windows available for given date range and config.")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = _hash_config(asdict(cfg))
    bar_delta = pd.Timedelta(minutes=15)
    purge_td = cfg.horizon_bars * bar_delta

    folds = []
    for fold_id, (tr_s, tr_e, va_s, va_e, te_s, te_e) in enumerate(windows):
        # Purge/embargo boundaries.
        train_mask = (t_idx >= tr_s) & (t_idx < (va_s - purge_td))
        val_mask = (t_idx >= (va_s + purge_td)) & (t_idx < (te_s - purge_td))
        test_mask = (t_idx >= (te_s + purge_td)) & (t_idx < te_e)
        if train_mask.sum() < 128 or val_mask.sum() < 32 or test_mask.sum() < 32:
            continue

        x_tr_seq, x_va_seq, x_te_seq = seq_arr[train_mask], seq_arr[val_mask], seq_arr[test_mask]
        x_tr_ctx, x_va_ctx, x_te_ctx = ctx_arr[train_mask], ctx_arr[val_mask], ctx_arr[test_mask]
        y_tr_reg, y_va_reg, y_te_reg = y_regime[train_mask].to_numpy(), y_regime[val_mask].to_numpy(), y_regime[test_mask].to_numpy()

        # 1) Regime model.
        reg = RegimeMLP(ctx_dim=x_tr_ctx.shape[1], hidden=32, out=4, dropout=0.1)
        reg_stats = train_regime_mlp(reg, x_tr_ctx, y_tr_reg, epochs=20, lr=1e-3)
        p_tr_reg = predict_regime_proba(reg, x_tr_ctx)
        p_va_reg = predict_regime_proba(reg, x_va_ctx)
        p_te_reg = predict_regime_proba(reg, x_te_ctx)

        # 2) Opportunity + Direction.
        y_tr_opp = aligned.loc[t_idx[train_mask], "y_opportunity"].to_numpy(dtype=np.float32)
        y_va_opp = aligned.loc[t_idx[val_mask], "y_opportunity"].to_numpy(dtype=np.float32)
        y_te_opp = aligned.loc[t_idx[test_mask], "y_opportunity"].to_numpy(dtype=np.float32)
        y_tr_dir = aligned.loc[t_idx[train_mask], "y_direction"].fillna(-1).to_numpy(dtype=np.int64)
        y_va_dir = aligned.loc[t_idx[val_mask], "y_direction"].fillna(-1).to_numpy(dtype=np.int64)
        y_te_dir = aligned.loc[t_idx[test_mask], "y_direction"].fillna(-1).to_numpy(dtype=np.int64)

        opp = OpportunityModel(seq_features=x_tr_seq.shape[2], ctx_dim=x_tr_ctx.shape[1], regime_dim=4)
        opp_stats = train_opportunity_model(opp, x_tr_seq, x_tr_ctx, p_tr_reg, y_tr_opp, epochs=15, lr=1e-3)
        opp.eval()
        with torch.no_grad():
            p_te_trade = opp(
                torch.as_tensor(x_te_seq, dtype=torch.float32),
                torch.as_tensor(x_te_ctx, dtype=torch.float32),
                torch.as_tensor(p_te_reg, dtype=torch.float32),
            ).detach().cpu().numpy()

        direc = DirectionModel(seq_features=x_tr_seq.shape[2], ctx_dim=x_tr_ctx.shape[1], regime_dim=4)
        dir_stats = train_direction_model(direc, x_tr_seq, x_tr_ctx, p_tr_reg, y_tr_dir, epochs=15, lr=1e-3)
        direc.eval()
        with torch.no_grad():
            p_te_dir = direc(
                torch.as_tensor(x_te_seq, dtype=torch.float32),
                torch.as_tensor(x_te_ctx, dtype=torch.float32),
                torch.as_tensor(p_te_reg, dtype=torch.float32),
            ).detach().cpu().numpy()

        # 3) Specialists.
        y_tr_break = aligned.loc[t_idx[train_mask], "y_breakout"].fillna(0.0).to_numpy(dtype=np.float32)
        y_tr_mean = aligned.loc[t_idx[train_mask], "y_mean_reversion"].fillna(0.0).to_numpy(dtype=np.float32)
        brk = BreakoutModel(seq_features=x_tr_seq.shape[2])
        brk_stats = train_breakout_model(brk, x_tr_seq, y_tr_break, epochs=10, lr=1e-3)
        mr = MeanReversionModel(seq_features=x_tr_seq.shape[2])
        mr_stats = train_mean_reversion_model(mr, x_tr_seq, y_tr_mean, epochs=10, lr=1e-3)
        brk.eval()
        mr.eval()
        with torch.no_grad():
            p_te_break = brk(torch.as_tensor(x_te_seq, dtype=torch.float32)).detach().cpu().numpy()
            p_te_mean = mr(torch.as_tensor(x_te_seq, dtype=torch.float32)).detach().cpu().numpy()

        # 4) Risk.
        x_tr_risk = np.concatenate([x_tr_ctx, p_tr_reg], axis=1)
        x_te_risk = np.concatenate([x_te_ctx, p_te_reg], axis=1)
        y_tr_risk = aligned.loc[t_idx[train_mask], "y_risk"].to_numpy(dtype=np.float32)
        risk = RiskModel(in_dim=x_tr_risk.shape[1], dropout=0.1)
        risk_stats = train_risk_model(risk, x_tr_risk, y_tr_risk, epochs=15, lr=1e-3)
        risk.eval()
        with torch.no_grad():
            sigma_hat = risk(torch.as_tensor(x_te_risk, dtype=torch.float32)).detach().cpu().numpy()

        # Probability signals for backtester/evaluator interface.
        prob_df = pd.DataFrame(
            {
                "p_trade": p_te_trade.astype(float),
                "p_long": p_te_dir[:, 0].astype(float),
                "p_short": p_te_dir[:, 1].astype(float),
                "p_breakout": p_te_break.astype(float),
                "p_meanrev": p_te_mean.astype(float),
                "regime_trend": p_te_reg[:, 0].astype(float),
                "regime_range": p_te_reg[:, 1].astype(float),
                "regime_highvol": p_te_reg[:, 2].astype(float),
                "regime_lowliq": p_te_reg[:, 3].astype(float),
                "sigma_hat": sigma_hat.astype(float),
            },
            index=t_idx[test_mask],
        )
        eval_metrics = evaluator(prob_df)

        # Save fold artifacts.
        fold_dir = out_dir / f"fold_{fold_id:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(reg.state_dict(), fold_dir / "regime.pt")
        torch.save(opp.state_dict(), fold_dir / "opportunity.pt")
        torch.save(direc.state_dict(), fold_dir / "direction.pt")
        torch.save(brk.state_dict(), fold_dir / "breakout.pt")
        torch.save(mr.state_dict(), fold_dir / "mean_reversion.pt")
        torch.save(risk.state_dict(), fold_dir / "risk.pt")
        prob_df.to_parquet(fold_dir / "prob_signals.parquet")

        manifest = {
            "fold_id": fold_id,
            "window": {
                "train_start": str(tr_s),
                "train_end": str(tr_e),
                "val_start": str(va_s),
                "val_end": str(va_e),
                "test_start": str(te_s),
                "test_end": str(te_e),
            },
            "config_hash": cfg_hash,
            "feature_schema_version": cfg.feature_schema_version,
            "cost_model_version": cfg.cost_model_version,
            "config": asdict(cfg),
            "stats": {
                "regime": reg_stats,
                "opportunity": opp_stats,
                "direction": dir_stats,
                "breakout": brk_stats,
                "mean_reversion": mr_stats,
                "risk": risk_stats,
            },
            "metrics": eval_metrics,
            "n_samples": {
                "train": int(train_mask.sum()),
                "val": int(val_mask.sum()),
                "test": int(test_mask.sum()),
            },
            "schema": {
                "seq_len": int(fb.seq_len),
                "seq_features": int(x_tr_seq.shape[2]),
                "ctx_dim": int(x_tr_ctx.shape[1]),
                "risk_in_dim": int(x_tr_risk.shape[1]),
            },
        }
        (fold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        folds.append(manifest)

    summary = {
        "config_hash": cfg_hash,
        "folds_trained": len(folds),
        "output_dir": str(out_dir),
        "feature_schema_version": cfg.feature_schema_version,
        "cost_model_version": cfg.cost_model_version,
        "fold_metrics": [f.get("metrics", {}) for f in folds],
    }
    (Path(cfg.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
