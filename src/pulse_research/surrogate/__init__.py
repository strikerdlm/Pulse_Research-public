"""Surrogate models for the CGEM-Pulse coupling."""
from pulse_research.surrogate.gp import GPModel, train_gp
from pulse_research.surrogate.mfgp import MFGPModel, train_mfgp
from pulse_research.surrogate.types import SurrogateProtocol

__all__ = ["GPModel", "MFGPModel", "SurrogateProtocol", "train_gp", "train_mfgp"]
