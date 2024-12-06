from .efficient_kan import EfficientKANLinear, EfficientKAN
from .fast_kan import FastKANLayer, FastKAN, AttentionWithFastKANTransform
from .faster_kan import FasterKAN
from .bsrbf_kan import BSRBF_KAN
from .gottlieb_kan import GottliebKAN
from .mlp import MLP
from .fc_kan import FC_KAN
from .skan import SKAN
from .prkan import PRKAN

#from .functions import lsin, lcos, larctan, lrelu, lleaky_relu, lswish, lmish, lsoftplus, lhard_sigmoid, lelu, lshifted_softplus, lgelup
# "lsin", "lcos", "larctan", "lrelu", "lleaky_relu", "lswish", "lmish", "lsoftplus", "lhard_sigmoid", "lelu", "lshifted_softplus", "lgelup"

__all__ = ["EfficientKAN", "EfficientKANLinear", "FastKAN", "FasterKAN", "BSRBF_KAN", "GottliebKAN", "MLP", "FC_KAN", "SKAN", "PRKAN"]

