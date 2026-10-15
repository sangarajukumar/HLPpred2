import torch
import torch.nn.functional as F
import lightning as pl
from .architecture import Hemopi_arch, Hemopi_Regression_arch

class Hemopi_Module(pl.LightningModule):
    """
    Lightning module for HLPpred 2.0 binary classification.
    """
    def __init__(self, model_config, trainer_config):
        super(Hemopi_Module, self).__init__()
        self.model = Hemopi_arch(**vars(model_config))
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.trainer_config = trainer_config
    
    def forward(self, X, masks_X, X_probs, X_merci):
        return self.model(X, masks_X, X_probs, X_merci)


class Hemopi_Regression_Module(pl.LightningModule):
    """
    Lightning module for HLPpred 2.0 potency regression (pHC50).
    """
    def __init__(self, model_config, trainer_config):
        super(Hemopi_Regression_Module, self).__init__()
        self.model = Hemopi_Regression_arch(**vars(model_config))
        self.loss_fn = torch.nn.MSELoss()
        self.trainer_config = trainer_config
    
    def forward(self, X, masks_X, X_probs, X_merci):
        return self.model(X, masks_X, X_probs, X_merci)
