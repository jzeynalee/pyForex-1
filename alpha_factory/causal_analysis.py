"""alpha_factory/causal_analysis.py

Compatibility wrapper.

The Alpha Factory causality pipeline has been consolidated into
`alpha_factory.enhancements` for readability and to prioritize non-linear
relationships. Granger-based causality has been removed from the active path.

Keep importing from here if existing code relies on it.
"""

from .enhancements import compute_causality, get_top_causal_features, create_causal_network



__all__ = [
    'compute_causality',
    'get_top_causal_features',
    'create_causal_network',
]
