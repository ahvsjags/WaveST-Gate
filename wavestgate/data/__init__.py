"""Data utilities for WaveST-Gate."""

from wavestgate.data.assemble import BatchMetadata, assemble_wavestgate_batch
from wavestgate.data.build_spatial_graph import build_knn_graph, build_radius_graph
from wavestgate.data.extract_he_patches import extract_spot_patches
from wavestgate.data.ground_truth import count_cells_in_spots, proportions_from_counts
from wavestgate.data.preprocess_scrna import ReferencePrototypes, build_reference_prototypes, load_reference_prototypes_table
from wavestgate.data.preprocess_st import SpotExpressionTable, attach_coordinates, load_spot_coordinates, load_spot_expression
from wavestgate.data.synthetic import make_synthetic_batch

__all__ = [
    "BatchMetadata",
    "ReferencePrototypes",
    "SpotExpressionTable",
    "assemble_wavestgate_batch",
    "attach_coordinates",
    "build_knn_graph",
    "build_radius_graph",
    "build_reference_prototypes",
    "count_cells_in_spots",
    "extract_spot_patches",
    "load_spot_coordinates",
    "load_spot_expression",
    "load_reference_prototypes_table",
    "make_synthetic_batch",
    "proportions_from_counts",
]
