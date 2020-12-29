from utils.monitor_base import MonitorBase
from time import time
import torch

class SteamMonitor(MonitorBase):
    def __init__(self, model, valid_loader, config):
        super().__init__(model, valid_loader, config)

    def validation(self):
        """This function will compute loss, median errors, KITTI metrics, and draw visualizations."""
        raise NotImplementedError('Subclasses must override validation()!')