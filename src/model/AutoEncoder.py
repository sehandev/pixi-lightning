import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.type import BatchType


class LitAutoEncoder(L.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
        )

    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
    ):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
    ):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
        )
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 0.95**epoch,
            ),
            "name": "learning_rate",
        }
        return [optimizer], [lr_scheduler]
