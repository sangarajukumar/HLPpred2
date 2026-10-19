import torch
import torch.nn.functional as F
import lightning as pl
from .architecture import Hemopi_arch
from .metrics import calculate_metrics
import numpy as np

class Hemopi_Module(pl.LightningModule):
    def __init__(self, model_config, trainer_config):
        super(Hemopi_Module, self).__init__()
        self.model = Hemopi_arch(**vars(model_config))
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.trainer_config = trainer_config

    def forward(self, X, masks_X, X_probs, X_merci):
        return self.model(X, masks_X, X_probs, X_merci)

    def training_step(self, batch, batch_idx):
        X, X_probs, X_merci, masks_X, y = batch
        y_logits = self.forward(X, masks_X, X_probs, X_merci)
        loss = self.loss_fn(y_logits, y)
        self.log('train_loss', loss)
        return loss

    def on_validation_epoch_start(self):
        self.all_y = []
        self.all_y_hat = []
        self.all_y_probs = []

    def validation_step(self, batch, batch_idx):
        X, X_probs, X_merci, masks_X, y = batch
        y_logits = self.forward(X, masks_X, X_probs, X_merci)
        y_probs = F.softmax(y_logits, dim=-1)

        y_probs_reversed = y_probs[:, 1]

        self.all_y.append(y)
        self.all_y_probs.append(y_probs_reversed)

    def on_validation_epoch_end(self):
        all_y = torch.cat(self.all_y).detach().cpu().numpy()
        all_y_probs = torch.cat(self.all_y_probs).detach().cpu().numpy()

        metrics = calculate_metrics(all_y_probs, all_y, threshold=0.5)

        self.log('sn', metrics['sn'], prog_bar=True)
        self.log('sp', metrics['sp'], prog_bar=True)
        self.log('fdr', metrics['fdr'], prog_bar=False)
        self.log('acc', metrics['acc'], prog_bar=True)
        self.log('bacc', metrics['bacc'], prog_bar=True)
        self.log('f1', metrics['f1'], prog_bar=True)
        self.log('mcc', metrics['mcc'], prog_bar=True)
        self.log('auc_roc', metrics['auc_roc'], prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.trainer_config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        return [optimizer], [scheduler]
