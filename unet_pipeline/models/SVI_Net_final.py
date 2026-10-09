"""SVI-Net uses the same BoxNet class; pickle may look for this module name."""

from models.EPE_Net_final import BoxNet, sigmoid_acc

__all__ = ["BoxNet", "sigmoid_acc"]
