"""gateway: the only publicly routable service. Composition and narrative."""

from .narrative import explain, headline, significance_banner
from .state import GatewayState, build_state_from_training, get_state, set_state

__version__ = "2.0.0"

__all__ = [
    "GatewayState",
    "build_state_from_training",
    "explain",
    "get_state",
    "headline",
    "set_state",
    "significance_banner",
]
