"""
Variant factory — builds all 6 experimental configurations.
"""

from typing import List

from ..interfaces import NullMHTCNFilter, VariantConfig, VariantID
from ..alpha_heads import AlphaHeadV1, AlphaHeadV2
from ..mhtcn_filters import RawFeatureMHTCNFilter, ProbabilisticMHTCNFilter


def build_all_variants(
    raw_mhtcn_weights: str = "",
    prob_mhtcn_weights: str = "",
    device: str = "cpu",
    initial_balance: float = 10_000.0,
    risk_per_trade: float = 0.01,
    commission_per_lot: float = 7.0,
    spread_pips: float = 1.0,
    pip_value: float = 10.0,
    min_rr: float = 1.5,
    atr_sl_mult: float = 2.0,
    max_open_trades: int = 1,
    cooldown_bars: int = 6,
    min_probability: float = 0.55,
) -> List[VariantConfig]:
    """Construct all 6 variant configs with shared execution params."""

    kw = dict(
        initial_balance=initial_balance, risk_per_trade=risk_per_trade,
        commission_per_lot=commission_per_lot, spread_pips=spread_pips,
        pip_value=pip_value, min_rr=min_rr, atr_sl_mult=atr_sl_mult,
        max_open_trades=max_open_trades, cooldown_bars=cooldown_bars,
        min_probability=min_probability,
    )

    a1 = AlphaHeadV1()
    a2 = AlphaHeadV2()
    nf = NullMHTCNFilter()
    rf = RawFeatureMHTCNFilter(weights_path=raw_mhtcn_weights or None, device=device)
    pf = ProbabilisticMHTCNFilter(weights_path=prob_mhtcn_weights or None, device=device)

    return [
        VariantConfig(VariantID.V1_ALPHA, a1, nf, "Alpha only — baseline", **kw),
        VariantConfig(VariantID.V2_ALPHA_MHTCN, a1, rf, "Alpha + raw MH-TCN", **kw),
        VariantConfig(VariantID.V3_ALPHA2, a2, nf, "Alpha2 only", **kw),
        VariantConfig(VariantID.V4_ALPHA2_MHTCN, a2, rf, "Alpha2 + raw MH-TCN", **kw),
        VariantConfig(VariantID.V5_ALPHA_PROB_MHTCN, a1, pf, "Alpha + prob MH-TCN", **kw),
        VariantConfig(VariantID.V6_ALPHA2_PROB_MHTCN, a2, pf, "Alpha2 + prob MH-TCN", **kw),
    ]
