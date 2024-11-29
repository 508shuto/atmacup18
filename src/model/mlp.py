import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup


# define a class for the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.model(x)


class MLPModel(pl.LightningModule):
    def __init__(
        self, config, input_dim, output_dim, num_warmup_steps, num_training_steps
    ):
        super().__init__()
        self.config = config
        self.backbone = MLP(input_dim, output_dim)
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # == pred ==
        with torch.no_grad():
            y_pred = self(x)

        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)

    def configure_optimizers(self):
        # == define optimizer ==
        model_optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # == define learning rate scheduler ==
        lr_scheduler = get_cosine_schedule_with_warmup(
            model_optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        return {
            "optimizer": model_optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    config = {}
    input_dim = 100
    output_dim = 18
    model = MLP(input_dim, output_dim)
    input = torch.zeros(2, input_dim)
    output = model(input)
    print(output.shape)
    print(output)
