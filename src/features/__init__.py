"""
Feature extraction module for polymer-specific properties
"""

from .polymer_features import (
    PolymerFeatureExtractor,
    calculate_molecular_weight,
    encode_degree_polymerization,
    extract_polymer_features,
    extract_repetition_unit_features,
)

__all__ = [
    "calculate_molecular_weight",
    "encode_degree_polymerization",
    "extract_repetition_unit_features",
    "extract_polymer_features",
    "PolymerFeatureExtractor",
]
