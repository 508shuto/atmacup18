import numpy as np
import torch
import wandb
from pytorch_lightning import Trainer as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

from src.dataset import MLPDataset
from src.model import MLPModel


class NNTrainer:
    def __init__(self, config):
        self.config = config

    def train(self, data_loader): ...

    def one_fold_train(self, fold_id, total_df):
        print("================================================================")
        print(f"==== Running training for fold {fold_id} ====")

        # == create dataset and dataloader ==
        train_df = total_df[total_df["fold"] != fold_id].copy()
        valid_df = total_df[total_df["fold"] == fold_id].copy()

        print(f"Train Samples: {len(train_df)}")
        print(f"Valid Samples: {len(valid_df)}")

        train_dataset = MLPDataset(train_X, train_y)
        val_dataset = MLPDataset(valid_X, valid_y)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config["train_batch_size"],
            shuffle=True,
            num_workers=self.config["n_workers"],
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config["valid_batch_size"],
            shuffle=False,
            num_workers=self.config["n_workers"],
            pin_memory=True,
            persistent_workers=True,
        )

        num_warmup_steps = len(train_dataloader)
        num_training_steps = len(train_dataloader) * self.config["epochs"]

        # == init model ==
        model = MLPModel(
            model_name=self.config["model_name"],
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            pretrained=True,
        )

        # == init callback ==
        checkpoint_callback = ModelCheckpoint(
            monitor="val_score",
            dirpath=self.config["output_dir"],
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
            filename=f"fold_{fold_id}",
            mode="max",
        )

        early_stopping = EarlyStopping(
            monitor="val_score",
            min_delta=0.00,
            patience=self.config["patience"],
            verbose=False,
            mode="max",
        )

        callbacks_to_use = [
            checkpoint_callback,
            early_stopping,
            TQDMProgressBar(refresh_rate=1),
        ]

        # == init logger ==
        wandb_logger = WandbLogger(
            project="atmacup18",
            name=f"{model.loss_fn.__class__.__name__}_fold{fold_id:02d}",
            config=self.config,
        )

        # == init trainer ==
        trainer = L.Trainer(
            max_epochs=self.config["epochs"],
            val_check_interval=0.5,
            callbacks=callbacks_to_use,
            enable_model_summary=False,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            deterministic=True,
            precision="16-mixed" if self.config["mixed_precision"] else 32,
            logger=wandb_logger,
        )

        # == Training ==
        trainer.fit(
            model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )
        wandb.finish()

        # == Prediction ==
        best_model_path = checkpoint_callback.best_model_path
        weights = torch.load(best_model_path)["state_dict"]
        model.load_state_dict(weights)

        preds, gts = self.predict(val_dataloader, model)

        # = compute score =
        val_score = self.evaluation(gts, preds)

        return preds, gts, val_score

    def predict(self, data_loader, model):
        model.to(self.config["device"])
        model.eval()
        predictions = []
        gts = []
        for batch in tqdm(data_loader):
            with torch.no_grad():
                x, y = batch
                x = x.cuda()
                outputs, _ = model(x)
            predictions.append(outputs.detach().cpu())
            gts.append(y.detach().cpu())

        predictions = torch.cat(predictions, dim=0).cpu().detach()
        gts = torch.cat(gts, dim=0).cpu().detach()

        return predictions.numpy().astype(np.float32), gts.numpy().astype(np.float32)

    def evaluation(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
