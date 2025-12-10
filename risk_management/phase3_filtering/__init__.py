"""
Phase 3: Trade Filtering

Components for filtering and validating trade signals:
- Triple Barrier Labeling: Creates supervised learning targets from trade outcomes
- Meta-Labeling: GBM model to predict P(primary_prediction_correct)
"""

from .triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabeler,
    TripleBarrierDataset,
    BarrierOutcome,
    BarrierLabel,
    create_triple_barrier_labels_from_model
)

from .meta_labeling import (
    MetaLabelingConfig,
    MetaLabelingModel,
    MetaFeatureExtractor,
    TradeFilter
)

__all__ = [
    # Triple Barrier
    'TripleBarrierConfig',
    'TripleBarrierLabeler',
    'TripleBarrierDataset',
    'BarrierOutcome',
    'BarrierLabel',
    'create_triple_barrier_labels_from_model',
    # Meta-Labeling
    'MetaLabelingConfig',
    'MetaLabelingModel',
    'MetaFeatureExtractor',
    'TradeFilter'
]
