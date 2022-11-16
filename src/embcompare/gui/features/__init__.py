try:
    import altair
    import loguru
    import pandas
    import streamlit
except ImportError:
    raise ImportError(
        "gui dependencies are not installed. "
        "Please install them by running : pip install embcompare[gui]"
    )

from .config_comparison import display_embeddings_config, display_numbers_of_elements
from .frequencies_comparison import display_frequencies_comparison
from .global_param_selection import display_parameters_selection
from .neighborhoods_comparison import (
    display_custom_elements_comparison,
    display_elements_comparison,
)
from .neighborhoods_sim_stats import display_neighborhoods_similarities
from .space_comparison import display_spaces_comparison
from .statistics_comparison import display_statistics_comparison

__all__ = [
    "display_parameters_selection",
    "display_embeddings_config",
    "display_numbers_of_elements",
    "display_statistics_comparison",
    "display_spaces_comparison",
    "display_neighborhoods_similarities",
    "display_elements_comparison",
    "display_custom_elements_comparison",
    "display_frequencies_comparison",
]
