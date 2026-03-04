#!/usr/bin/env python3
"""Run baseline multi-timeframe data prep, training, and holdout evaluation."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.manager import DataManager


def _parse_date(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def _add_months(value: dt.datetime, months: int) -> dt.datetime:
    month = value.month - 1 + months
    year = value.year + month // 12
    month = month % 12 + 1
    # Keep day in safe range.
    day = min(value.day, 28)
    return value.replace(year=year, month=month, day=day)


def _chunk_ranges(start: dt.datetime, end: dt.datetime, months: int) -> Iterable[Tuple[dt.datetime, dt.datetime]]:
    cur = start
    while cur < end:
        nxt = _add_months(cur, months)
        if nxt > end:
            nxt = end
        yield cur, nxt
        cur = nxt


def _run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=True)
    return proc.stdout


def _parse_runtime_metrics(output: str) -> dict:
    def grab(pattern: str) -> str:
        m = re.search(pattern, output)
        if not m:
            raise ValueError(f"Missing metric in runtime output: {pattern}")
        return m.group(1)

    return {
        "trades": int(grab(r"Trades: ([0-9]+)")),
        "final_equity": float(grab(r"Final equity: ([0-9.\-]+)")),
        "net_pnl": float(grab(r"Net PnL: ([0-9.\-]+)")),
        "win_rate_pct": float(grab(r"Win rate: ([0-9.]+)%")),
        "sharpe": float(grab(r"Sharpe: ([0-9.\-]+)")),
        "max_drawdown_pct": float(grab(r"Max drawdown: ([0-9.]+)%")),
        "fees": float(grab(r"Fees: ([0-9.\-]+)")),
        "financing": float(grab(r"Financing: ([0-9.\-]+)")),
    }


def _status_from_gate(metrics: dict) -> str:
    if (
        metrics["net_pnl"] > 0
        and metrics["max_drawdown_pct"] <= 20.0
        and metrics["trades"] >= 50
        and metrics["sharpe"] > 0
    ):
        return "PASS"
    return "REJECT"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline multi-timeframe training/eval pipeline.")
    p.add_argument(
        "--mode",
        choices=["prepare-data", "train", "eval", "full"],
        default="full",
        help="Which pipeline stage(s) to run.",
    )
    p.add_argument("--instruments", default="EUR_USD,GBP_USD,USD_JPY,XAU_USD")
    p.add_argument("--base-tf", default="M15", help="Execution-aligned base timeframe for model training.")
    p.add_argument("--htf-1", default="H1")
    p.add_argument("--htf-2", default="H4")
    p.add_argument("--prepare-extra-tfs", default="D1", help="Additional TFs to pre-cache, comma-separated.")
    p.add_argument("--chunk-months", type=int, default=3)
    p.add_argument("--fine-start", default="2022-01-01")
    p.add_argument("--fine-end", default="2024-12-31")
    p.add_argument("--full-start", default="2015-01-01")
    p.add_argument("--full-end", default="2024-12-31")
    p.add_argument("--oos-start", default="2025-01-01")
    p.add_argument("--oos-end", default="2025-12-31")
    p.add_argument("--eval-tfs", default="H1,M15", help="Runtime eval TFs (avoid M1 for this multiframe model).")
    p.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--strategy-params-csv", default="")
    p.add_argument("--risk-per-trade-pct", type=float, default=0.01)
    p.add_argument("--max-notional-exposure-pct", type=float, default=1.0)
    p.add_argument("--min-quantity", type=int, default=1)
    p.add_argument("--max-quantity", type=int, default=100000)
    p.add_argument("--max-drawdown-stop-pct", type=float, default=0.20)
    p.add_argument("--daily-loss-limit-pct", type=float, default=0.05)
    p.add_argument("--financing", choices=["on", "off"], default="on")
    p.add_argument("--default-financing-long-rate", type=float, default=0.03)
    p.add_argument("--default-financing-short-rate", type=float, default=0.03)
    p.add_argument("--rollover-hour-utc", type=int, default=22)
    p.add_argument("--output-dir", default="data/research")
    p.add_argument("--manifest-json", default="", help="Optional manifest path for eval-only mode.")
    return p.parse_args()


def _split_csv(values: str) -> List[str]:
    return [x.strip() for x in values.split(",") if x.strip()]


def _to_tf_list(values: str) -> List[Timeframe]:
    return [Timeframe.from_oanda_granularity(v) for v in _split_csv(values)]


def _prepare_data(
    instruments: List[str],
    tfs: List[Timeframe],
    start: dt.datetime,
    end: dt.datetime,
    chunk_months: int,
) -> None:
    dm = DataManager({})
    base_tf = tfs[0]
    for inst in instruments:
        for cstart, cend in _chunk_ranges(start, end, chunk_months):
            dm.ensure_data(
                instrument=inst,
                base_timeframe=base_tf,
                start_date=cstart,
                end_date=cend,
                timeframes=tfs,
                force_download=False,
            )
            print(f"Prepared data: {inst} {base_tf.name} {cstart.date()} -> {cend.date()}")


def _train_multiframe(
    instruments: List[str],
    base_tf: str,
    htf_1: str,
    htf_2: str,
    start: str,
    end: str,
    gpu: str,
    output_dir: str,
) -> str:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_multiframe_regime_model.py"),
        "--instruments",
        ",".join(instruments),
        "--base-tf",
        base_tf,
        "--htf-1",
        htf_1,
        "--htf-2",
        htf_2,
        "--start",
        start,
        "--end",
        end,
        "--gpu",
        gpu,
        "--output-dir",
        output_dir,
    ]
    out = _run_cmd(cmd)
    m = re.search(r"Model JSON: (.+)", out)
    if not m:
        raise RuntimeError(f"Could not parse model path from output:\n{out}")
    model_path = m.group(1).strip()
    print(f"Trained multiframe model: {model_path}")
    return model_path


def _evaluate(
    instruments: List[str],
    eval_tfs: List[str],
    model_json: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / f"mtf_baseline_eval_{stamp}.csv"
    headers = [
        "instrument",
        "tf",
        "oos_start",
        "oos_end",
        "trades",
        "final_equity",
        "net_pnl",
        "win_rate_pct",
        "sharpe",
        "max_drawdown_pct",
        "fees",
        "financing",
        "status",
        "model_json",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for inst in instruments:
            for tf in eval_tfs:
                cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "run_regime_runtime_backtest.py"),
                    "--model-json",
                    model_json,
                    "--instrument",
                    inst,
                    "--tf",
                    tf,
                    "--start",
                    args.oos_start,
                    "--end",
                    args.oos_end,
                    "--fill-mode",
                    "next_open",
                    "--decision-mode",
                    "ensemble",
                    "--risk-per-trade-pct",
                    str(args.risk_per_trade_pct),
                    "--max-notional-exposure-pct",
                    str(args.max_notional_exposure_pct),
                    "--min-quantity",
                    str(args.min_quantity),
                    "--max-quantity",
                    str(args.max_quantity),
                    "--max-drawdown-stop-pct",
                    str(args.max_drawdown_stop_pct),
                    "--daily-loss-limit-pct",
                    str(args.daily_loss_limit_pct),
                    "--financing",
                    args.financing,
                    "--default-financing-long-rate",
                    str(args.default_financing_long_rate),
                    "--default-financing-short-rate",
                    str(args.default_financing_short_rate),
                    "--rollover-hour-utc",
                    str(args.rollover_hour_utc),
                ]
                if args.strategy_params_csv:
                    cmd.extend(["--strategy-params-csv", args.strategy_params_csv])
                out = _run_cmd(cmd)
                metrics = _parse_runtime_metrics(out)
                status = _status_from_gate(metrics)
                writer.writerow(
                    {
                        "instrument": inst,
                        "tf": tf,
                        "oos_start": args.oos_start,
                        "oos_end": args.oos_end,
                        "trades": metrics["trades"],
                        "final_equity": f"{metrics['final_equity']:.2f}",
                        "net_pnl": f"{metrics['net_pnl']:.2f}",
                        "win_rate_pct": f"{metrics['win_rate_pct']:.2f}",
                        "sharpe": f"{metrics['sharpe']:.4f}",
                        "max_drawdown_pct": f"{metrics['max_drawdown_pct']:.2f}",
                        "fees": f"{metrics['fees']:.2f}",
                        "financing": f"{metrics['financing']:.2f}",
                        "status": status,
                        "model_json": model_json,
                    }
                )
                print(
                    f"Eval {inst} {tf}: status={status} pnl={metrics['net_pnl']:.2f} "
                    f"sharpe={metrics['sharpe']:.4f} dd={metrics['max_drawdown_pct']:.2f}%"
                )
    return str(out_csv)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    instruments = _split_csv(args.instruments)
    eval_tfs = _split_csv(args.eval_tfs)

    base_tf = Timeframe.from_oanda_granularity(args.base_tf)
    htf_1 = Timeframe.from_oanda_granularity(args.htf_1)
    htf_2 = Timeframe.from_oanda_granularity(args.htf_2)
    extra_tfs = _to_tf_list(args.prepare_extra_tfs) if args.prepare_extra_tfs else []
    prep_tfs = [base_tf, htf_1, htf_2] + [x for x in extra_tfs if x not in {base_tf, htf_1, htf_2}]

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest_path = Path(args.manifest_json) if args.manifest_json else output_dir / f"mtf_baseline_manifest_{stamp}.json"
    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": args.mode,
        "instruments": instruments,
        "base_tf": args.base_tf,
        "htf_1": args.htf_1,
        "htf_2": args.htf_2,
        "fine_window": {"start": args.fine_start, "end": args.fine_end},
        "full_window": {"start": args.full_start, "end": args.full_end},
        "oos_window": {"start": args.oos_start, "end": args.oos_end},
        "models": {},
        "outputs": {},
    }

    if args.mode in {"prepare-data", "full"}:
        _prepare_data(
            instruments=instruments,
            tfs=prep_tfs,
            start=_parse_date(args.full_start),
            end=_parse_date(args.oos_end),
            chunk_months=args.chunk_months,
        )
        manifest["outputs"]["prepared_tfs"] = [tf.name for tf in prep_tfs]

    if args.mode in {"train", "full"}:
        fine_model = _train_multiframe(
            instruments=instruments,
            base_tf=args.base_tf,
            htf_1=args.htf_1,
            htf_2=args.htf_2,
            start=args.fine_start,
            end=args.fine_end,
            gpu=args.gpu,
            output_dir=args.output_dir,
        )
        full_model = _train_multiframe(
            instruments=instruments,
            base_tf=args.base_tf,
            htf_1=args.htf_1,
            htf_2=args.htf_2,
            start=args.full_start,
            end=args.full_end,
            gpu=args.gpu,
            output_dir=args.output_dir,
        )
        manifest["models"]["fine_tuned"] = fine_model
        manifest["models"]["full_history"] = full_model

    if args.mode == "eval":
        if args.manifest_json:
            loaded = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
            model_json = str(loaded.get("models", {}).get("full_history", ""))
            if not model_json:
                raise SystemExit("Manifest does not contain models.full_history")
            manifest.update(loaded)
        else:
            candidates = sorted(output_dir.glob("multiframe_regime_model_*.json"))
            if not candidates:
                raise SystemExit("No multiframe model found for eval mode.")
            model_json = str(candidates[-1])
            manifest["models"]["full_history"] = model_json
        eval_csv = _evaluate(instruments, eval_tfs, model_json, args, output_dir)
        manifest["outputs"]["eval_csv"] = eval_csv

    if args.mode == "full":
        model_json = str(manifest["models"]["full_history"])
        eval_csv = _evaluate(instruments, eval_tfs, model_json, args, output_dir)
        manifest["outputs"]["eval_csv"] = eval_csv

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
