"""
Multi-Alpha + MH-TCN Research Architecture
==========================================

Controlled research framework to implement, backtest, and compare
six alpha + MH-TCN architecture variants.

Variants:
    V1: Alpha (existing Alpha Factory)
    V2: Alpha + MH-TCN (raw features)
    V3: Alpha2 (category-based probabilistic)
    V4: Alpha2 + MH-TCN (raw features)
    V5: Alpha + probabilistic MH-TCN
    V6: Alpha2 + probabilistic MH-TCN

Hard constraints:
    - Direction authority comes ONLY from Alpha heads
    - MH-TCN modulates probability only: P_final = P_alpha * g_mhtcn
    - No label leakage (MH-TCN never sees realized PnL)
    - Identical execution, risk, and costs across all variants
"""

__version__ = "1.0.0"
