#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import pandas as pd
from oandapyV20 import API
from oandapyV20.endpoints.accounts import AccountSummary

from scripts.promote_and_validate_selected_models import PROJECT_ROOT
from shared.config import Config


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    print(f"[shadow-reporter] {_now_iso()} {msg}", flush=True)


def _latest_heartbeat_csv(root: Path) -> Optional[Path]:
    files = sorted(root.glob("shadow_*/heartbeat_metrics.csv"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_shadow_snapshot(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        return {"status": "no_csv"}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"status": "empty_csv"}
    last_ts = str(df["ts_utc"].iloc[-1])
    cur = df[df["ts_utc"] == last_ts].copy()
    for c in ["trades", "net_expectancy_after_cost", "profit_factor", "max_dd"]:
        if c in cur.columns:
            cur[c] = pd.to_numeric(cur[c], errors="coerce")
    ok = cur[cur["status"] == "ok"].copy()
    if ok.empty:
        return {"status": "no_ok_rows", "last_ts": last_ts}
    w = ok["trades"].fillna(0.0).clip(lower=0.0)
    denom = float(max(1.0, float(w.sum())))
    weighted_net_exp = float((ok["net_expectancy_after_cost"].fillna(0.0) * w).sum() / denom)
    profitable = ok[(ok["net_expectancy_after_cost"] > 0.0) & (ok["profit_factor"] > 1.0)]
    worst_dd = float(pd.to_numeric(ok["max_dd"], errors="coerce").fillna(0.0).max()) if "max_dd" in ok.columns else 0.0
    return {
        "status": "ok",
        "last_ts": last_ts,
        "rows_ok": int(len(ok)),
        "rows_profitable": int(len(profitable)),
        "weighted_net_expectancy_after_cost": weighted_net_exp,
        "worst_max_dd": worst_dd,
        "top_profitable_rows": profitable[
            ["symbol", "model_path", "trades", "net_expectancy_after_cost", "profit_factor"]
        ].head(5).to_dict(orient="records"),
    }


def _fetch_balance_nav(api: API, account_id: str) -> Dict[str, Any]:
    req = AccountSummary(accountID=account_id)
    res = api.request(req)
    acct = res.get("account", {}) if isinstance(res, dict) else {}
    bal = float(acct.get("balance", 0.0))
    nav = float(acct.get("NAV", bal))
    return {"balance": bal, "nav": nav}


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _build_report_text(acct: Dict[str, Any], state: Dict[str, Any], shadow: Dict[str, Any], interval_h: float) -> str:
    nav = acct.get("nav")
    bal = acct.get("balance")
    if nav is None or bal is None:
        nav = float(state.get("last_nav", 0.0))
        bal = float(state.get("last_balance", 0.0))
    nav = float(nav)
    bal = float(bal)
    start_nav = float(state.get("start_nav", nav))
    peak_nav = float(state.get("peak_nav", nav))
    pnl_cash = nav - start_nav
    pnl_pct = (pnl_cash / start_nav * 100.0) if start_nav > 0 else 0.0
    dd_pct = ((peak_nav - nav) / peak_nav * 100.0) if peak_nav > 0 else 0.0

    lines = [
        f"SHADOW REPORT ({interval_h:.1f}h)",
        f"- time_utc: `{_now_iso()}`",
        f"- account_balance: `{bal:.2f}`" if acct.get("balance") is not None else "- account_balance: `unavailable`",
        f"- account_nav(equity): `{nav:.2f}`" if acct.get("nav") is not None else "- account_nav(equity): `unavailable`",
        f"- pnl_since_reporter_start: `{pnl_cash:+.2f}` ({pnl_pct:+.2f}%)" if acct.get("nav") is not None else "- pnl_since_reporter_start: `unavailable`",
        f"- drawdown_from_peak: `{dd_pct:.2f}%`" if acct.get("nav") is not None else "- drawdown_from_peak: `unavailable`",
    ]
    if acct.get("fetch_error"):
        lines.append(f"- account_fetch_error: `{acct['fetch_error']}`")
    if shadow.get("status") == "ok":
        lines.extend(
            [
                f"- shadow_last_ts: `{shadow.get('last_ts')}`",
                f"- shadow_ok_models: `{shadow.get('rows_ok')}`",
                f"- shadow_profitable_models: `{shadow.get('rows_profitable')}`",
                f"- shadow_weighted_net_expectancy: `{float(shadow.get('weighted_net_expectancy_after_cost', 0.0)):+.6f}`",
                f"- shadow_worst_model_max_dd: `{float(shadow.get('worst_max_dd', 0.0)):.4f}`",
            ]
        )
    else:
        lines.append(f"- shadow_status: `{shadow.get('status', 'unknown')}`")
    return "\n".join(lines)


async def _send_discord_message(http: aiohttp.ClientSession, headers: Dict[str, str], channel_id: str, content: str) -> None:
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    payload = {"content": content[:1900]}
    async with http.post(url, headers=headers, json=payload) as resp:
        if resp.status >= 300:
            body = await resp.text()
            raise RuntimeError(f"Discord post failed: {resp.status} {body}")


async def amain() -> int:
    p = argparse.ArgumentParser(description="Periodic shadow performance reporter to Discord.")
    p.add_argument("--interval-seconds", type=int, default=7200)
    p.add_argument("--shadow-root", default="data/research/shadow_watch")
    p.add_argument("--state-path", default="data/research/shadow_watch/reporter_state.json")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    cfg = Config.load()
    token = (os.getenv("DISCORD_EXEC_BOT_TOKEN", "").strip() or os.getenv("DISCORD_BOT_TOKEN", "").strip())
    channel_id = (os.getenv("DISCORD_EXEC_CHANNEL_ID", "").strip() or os.getenv("DISCORD_CHANNEL_ID", "").strip())
    if not token or not channel_id:
        raise SystemExit("Missing DISCORD token/channel env vars")

    headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
    oanda_api = API(access_token=cfg.oanda.api_token, environment=cfg.oanda.environment)
    account_id = cfg.oanda.account_id
    shadow_root = PROJECT_ROOT / args.shadow_root
    state_path = PROJECT_ROOT / args.state_path

    _log(f"started interval={args.interval_seconds}s shadow_root={shadow_root}")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as http:
        while True:
            state = _load_state(state_path)
            try:
                acct = _fetch_balance_nav(oanda_api, account_id)
            except Exception as exc:
                acct = {"balance": None, "nav": None, "fetch_error": f"{type(exc).__name__}: {exc}"}
                _log(f"account fetch failed: {acct['fetch_error']}")
            nav_for_state = float(acct["nav"]) if acct.get("nav") is not None else float(state.get("last_nav", 0.0))
            bal_for_state = float(acct["balance"]) if acct.get("balance") is not None else float(state.get("last_balance", 0.0))
            start_nav = float(state.get("start_nav", nav_for_state))
            peak_nav = max(float(state.get("peak_nav", nav_for_state)), nav_for_state)
            state.update(
                {
                    "start_nav": start_nav,
                    "peak_nav": peak_nav,
                    "last_nav": nav_for_state,
                    "last_balance": bal_for_state,
                    "last_report_ts": _now_iso(),
                }
            )
            _save_state(state_path, state)

            hb = _latest_heartbeat_csv(shadow_root)
            shadow = _load_shadow_snapshot(hb) if hb is not None else {"status": "no_heartbeat_file"}
            text = _build_report_text(acct, state, shadow, interval_h=float(args.interval_seconds) / 3600.0)
            try:
                await _send_discord_message(http, headers, channel_id, text)
                _log("report sent")
            except Exception as exc:
                _log(f"report send failed: {type(exc).__name__}: {exc}")

            if args.once:
                break
            await asyncio.sleep(max(300, int(args.interval_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
