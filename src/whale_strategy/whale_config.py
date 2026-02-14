"""
Shared whale detection configuration for backtest and live opportunity finding.

Change whale definition here; both backtest and find_whale_opportunities use it.

Whale modes:
- default: identify_polymarket_whales (mid_price_accuracy or volume_top10)
- volume_only: 95th percentile volume in market (rolling)
- surprise_only: volume whales with positive surprise (requires resolutions)
- unfavored_only: filter to underdog trades (BUY <=40c, SELL >=60c)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class WhaleConfig:
    """Whale detection parameters shared by backtest and live scanner."""

    # Mode: "default" | "volume_only" | "surprise_only"
    mode: str = "volume_only"

    # Volume-only: 95th percentile in market
    volume_percentile: float = 95.0

    # Unfavored filter: only underdog trades
    unfavored_only: bool = False
    unfavored_max_price: float = 0.40

    # Surprise-only: min surprise to qualify (requires resolutions)
    min_surprise: float = 0.0
    min_trades_for_surprise: int = 10

    # Performance filter (default: only follow whales with WR >= 50% and positive surprise)
    min_whale_wr: float = 0.50
    require_positive_surprise: bool = True

    # Default scoring when no resolution
    default_whale_score: float = 7.0
    default_whale_winrate: float = 0.5

    # Min trade size (USD) to count
    min_usd: float = 100.0

    # For default mode: identify_polymarket_whales params
    min_trades: int = 10
    min_volume: float = 1000.0

    @property
    def volume_only(self) -> bool:
        return self.mode == "volume_only"

    @property
    def surprise_only(self) -> bool:
        return self.mode == "surprise_only"

    def __repr__(self) -> str:
        parts = [f"mode={self.mode}"]
        if self.min_whale_wr > 0 or self.require_positive_surprise:
            parts.append(f"WR>={self.min_whale_wr*100:.0f}% surprise>0")
        if self.unfavored_only:
            parts.append(f"unfavored<={self.unfavored_max_price}")
        return f"WhaleConfig({', '.join(parts)})"


def load_whale_config(config_path: Optional[Path] = None) -> WhaleConfig:
    """Load WhaleConfig from YAML (default + local merge), with defaults."""
    cfg = WhaleConfig()
    root = Path(__file__).resolve().parents[2]
    default_path = root / "config" / "default.yaml"
    local_path = root / "config" / "local.yaml"

    data = {}
    if default_path.exists():
        try:
            with open(default_path) as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            pass
    if local_path.exists():
        try:
            with open(local_path) as f:
                local = yaml.safe_load(f) or {}
            if isinstance(local, dict) and isinstance(data, dict):
                for k, v in local.items():
                    if k in data and isinstance(data[k], dict) and isinstance(v, dict):
                        data[k] = {**data[k], **v}
                    else:
                        data[k] = v
        except Exception:
            pass

    ws = data.get("whale_strategy") or {}
    if isinstance(ws, dict):
        if "whale_mode" in ws:
            cfg.mode = str(ws["whale_mode"])
        if "volume_percentile" in ws:
            cfg.volume_percentile = float(ws["volume_percentile"])
        if "unfavored_only" in ws:
            cfg.unfavored_only = bool(ws["unfavored_only"])
        if "unfavored_max_price" in ws:
            cfg.unfavored_max_price = float(ws["unfavored_max_price"])
        if "min_trades" in ws:
            cfg.min_trades = int(ws["min_trades"])
        if "min_volume" in ws:
            cfg.min_volume = float(ws["min_volume"])
        if "min_usd" in ws:
            cfg.min_usd = float(ws["min_usd"])
        if "min_whale_wr" in ws:
            cfg.min_whale_wr = float(ws["min_whale_wr"])
        if "require_positive_surprise" in ws:
            cfg.require_positive_surprise = bool(ws["require_positive_surprise"])

    return cfg


# Default instance
DEFAULT_WHALE_CONFIG = WhaleConfig()
