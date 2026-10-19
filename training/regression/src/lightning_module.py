import torch
import torch.nn.functional as F
import lightning as pl
from .architecture import Hemopi_Regression_arch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np

class Hemopi_Regression_Module(pl.LightningModule):
    def __init__(self, model_config, trainer_config):
        super(Hemopi_Regression_Module, self).__init__()
        self.model = Hemopi_Regression_arch(**vars(model_config))
        self.loss_fn = torch.nn.HuberLoss()

        self.trainer_config = trainer_config

    def forward(self, X, masks_X, X_probs, X_merci):
        return self.model(X, masks_X, X_probs, X_merci)

    def training_step(self, batch, batch_idx):
        X, X_probs, X_merci, masks_X, phc50 = batch
        phc50_pred = self.forward(X, masks_X, X_probs, X_merci)
        loss = self.loss_fn(phc50_pred, phc50)
        self.log('train_loss', loss)
        return loss

    def on_validation_epoch_start(self):
        self.all_y_true = []
        self.all_y_pred = []

    def validation_step(self, batch, batch_idx):
        X, X_probs, X_merci, masks_X, phc50 = batch
        phc50_pred = self.forward(X, masks_X, X_probs, X_merci)

        self.all_y_true.append(phc50)
        self.all_y_pred.append(phc50_pred)

    def on_validation_epoch_end(self):
        all_y_true = torch.cat(self.all_y_true).detach().cpu().numpy()
        all_y_pred = torch.cat(self.all_y_pred).detach().cpu().numpy()

        mask = np.isfinite(all_y_true) & np.isfinite(all_y_pred)
        all_y_true = all_y_true[mask]
        all_y_pred = all_y_pred[mask]

        mse = mean_squared_error(all_y_true, all_y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_y_true, all_y_pred)
        r2 = r2_score(all_y_true, all_y_pred)

        try:
            pcc, _ = pearsonr(all_y_true, all_y_pred)
            if not np.isfinite(pcc):
                pcc = 0.0
        except:
            pcc = 0.0

        self.log('r2', r2, prog_bar=True)
        self.log('rmse', rmse, prog_bar=True)
        self.log('mae', mae, prog_bar=True)
        self.log('pcc', pcc, prog_bar=True)
        self.log('mse', mse, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.trainer_config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        return [optimizer], [scheduler]
