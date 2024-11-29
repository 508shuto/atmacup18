from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

# fmt: off
TARGET_COLS = [
    "x_0", "y_0", "z_0",
    "x_1", "y_1", "z_1",
    "x_2", "y_2", "z_2",
    "x_3", "y_3", "z_3",
    "x_4", "y_4", "z_4",
    "x_5", "y_5", "z_5",
]
# fmt: on


@dataclass
class Config:
    # general config
    project_name: str = "atmacup18"
    exp: str = "exp100"
    seed: int = 42
    debug: bool = False
    input_dir: Path = Path("/Users/gouyashuto/localrepository/atmacup18/input")
    output_dir: Path = Path("/Users/gouyashuto/localrepository/atmacup18/output")
    is_wandb: bool = False if not debug else True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # feature engineering config
    groups: str = "scene"
    n_folds: int = 5

    # dataloader config
    batch_size: int = 16
    num_workers: int = 2

    # training config
    img_size: int = 224
    origin_img_size: list[int] = [64, 128]
    epochs: int = 10 if not debug else 3
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu"
    n_devices: int = 1
    use_amp: bool = False
    is_wandb: bool = False if not debug else True
    mix_precision: str = "bf16" if torch.cuda.is_available() else "fp16"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    patience: int = 5 if not debug else 2
    val_check_interval: float = 1.0
    checkpoint_name: str = "fold{}_weight_checkpoint_best"

    # model config
    criterion: nn.Module = nn.L1Loss()
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2

    # timm config
    model_name: str = "resnet18d"
    is_pretrained: bool = True
    in_chans: int = 3

    @classmethod
    def set_config_from_argparse(cls) -> "Config":
        parser = ArgumentParser()
        for k, v in cls.__dataclass_fields__.items():
            parser.add_argument(f"--{k}", type=type(v.default), default=v.default)
        args = parser.parse_args()
        return cls(**vars(args))


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df["scene"] = df["ID"].apply(lambda x: x.split("_")[0])
    df["frame"] = df["ID"].apply(lambda x: x.split("_")[1]).astype(int)
    df = df.sort_values(by=["scene", "frame"])
    return df


def get_fold(cfg: "Config", train_df: pd.DataFrame) -> pd.DataFrame:
    kf = GroupKFold(n_splits=cfg.n_folds)
    groups = train_df[cfg.groups]

    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(kf.split(train_df, groups=groups)):
        train_df.loc[val_idx, "fold"] = fold
    return train_df


def get_data(cfg: "Config") -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(cfg.input_dir / "train_features.csv")
    test_df = pd.read_csv(cfg.input_dir / "test_features.csv")

    train_df = preprocess_features(train_df)
    test_df = preprocess_features(test_df)

    train_df = get_fold(cfg, train_df)

    return train_df, test_df


class CustomDataset(Dataset):
    def __init__(
        self,
        cfg: "Config",
        df: pd.DataFrame,
        transform: A.Compose | None = None,
        is_test: bool = False,
    ):
        self.cfg = cfg
        self.df = df
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        scene = self.df.iloc[idx]["scene"]
        frame = self.df.iloc[idx]["frame"]
        if not self.is_test:
            targets = self.df.iloc[idx][TARGET_COLS]

        images = []
        for window in range(6):
            image_dir = (
                self.cfg.input_dir / "images" / f"{scene}_{int(frame + window * 100)}"
            )
            if image_dir.exists():
                image = self.get_frame_images(image_dir)
            else:
                h, w = self.cfg.origin_img_size
                c = self.cfg.in_chans
                image = np.zeros((h, w * 3, c), dtype=np.uint8)

            if self.transform:
                image = self.transform(image=image)["image"]

            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))

            images.append(image)

        images = np.array(images)

        if self.is_test:
            return {
                "images": torch.tensor(images, dtype=torch.float),
            }
        else:
            return {
                "images": torch.tensor(images, dtype=torch.float),
                "targets": torch.tensor(targets, dtype=torch.float),
            }

    def get_frame_images(self, frame_image_dir: Path) -> np.ndarray:
        frame_images = []
        for t in ["image_t-1.0.png", "image_t-0.5.png", "image_t.png"]:
            image_path = frame_image_dir / t
            image = cv2.imread(str(image_path), cv2.COLOR_BGR2RGB)
            frame_images.append(image)
        return np.concatenate(frame_images, axis=1)  # (h, w * 3, c)


class Net(nn.Module):
    def __init__(self, cfg: "Config"):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.is_pretrained,
            in_chans=cfg.in_chans,
            num_classes=0,
        )
        self.n_features = self.backbone.num_features
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(cfg.lstm_hidden_size * 2, len(TARGET_COLS))

    def forward(self, x):
        batch_size, sequence_length, c, h, w = x.shape
        video_features = x.view(batch_size * sequence_length, c, h, w)
        video_features = self.backbone(video_features)
        video_features = video_features.view(batch_size, sequence_length, -1)
        feature = F.relu(video_features)
        x, _ = self.lstm(feature)
        x = self.fc(x)
        return x[:, 0, :]


class CustomModel(L.LightningModule):
    def __init__(
        self,
        cfg: "Config",
        num_training_steps: int | None = None,
        num_warmup_steps: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = Net(cfg)
        self.loss_fn = cfg.criterion
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        targets = batch["targets"]

        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, batch_size=images.size(0)
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        targets = batch["targets"]

        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.log(
            "val_loss", loss, prog_bar=True, logger=True, batch_size=images.size(0)
        )
        return loss

    def configure_optimizers(self):
        # == define optimizer ==
        model_optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
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


def train_fn(cfg: "Config", df: pd.DataFrame, fold: int) -> None:
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    train_transformers = get_transform(cfg, is_train=True)
    valid_transformers = get_transform(cfg, is_train=False)

    train_dataset = CustomDataset(cfg, train_df, train_transformers)
    valid_dataset = CustomDataset(cfg, valid_df, valid_transformers)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    num_warmup_steps = len(train_dataloader)
    num_training_steps = len(train_dataloader) * cfg.epochs

    model = CustomModel(cfg, num_training_steps, num_warmup_steps)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=cfg.output_dir / cfg.exp / "model",
        filename=cfg.checkpoint_name.format(fold),
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
        mode="min",
    )

    callbacks = [checkpoint_callback, early_stopping]

    if cfg.is_wandb:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            config=cfg,
            save_dir=cfg.output_dir / cfg.exp / "log",
            name=f"fold{fold}_{cfg.exp}_{cfg.model_name}",
        )
        wandb_logger.watch(model)

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        val_check_interval=cfg.val_check_interval,
        accelerator=cfg.accelerator,
        devices=cfg.n_devices,
        enable_progress_bar=True,
        callbacks=callbacks,
        enable_model_summary=True,
        deterministic=True,
        precision=cfg.mix_precision,
        logger=wandb_logger if cfg.is_wandb else None,
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    if cfg.is_wandb:
        wandb.finish()


def inference(cfg: "Config", model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    model.to(cfg.device)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(cfg.device)
            outputs = model(images)
            preds.append(outputs.detach().cpu())
        preds = torch.cat(preds, dim=0).cpu().detach().numpy().astype(np.float32)

    return preds


def get_transform(cfg: "Config", is_train: bool = True) -> A.Compose:
    if is_train:
        return A.Compose(
            [
                A.Resize(cfg.img_size, cfg.img_size, interpolation=cv2.INTER_CUBIC),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(cfg.img_size, cfg.img_size, interpolation=cv2.INTER_CUBIC),
            ]
        )


def predict_fn(cfg: "Config", df: pd.DataFrame, model_path: Path | str) -> np.ndarray:
    test_transformers = get_transform(cfg, is_train=False)
    test_dataset = CustomDataset(cfg, df, test_transformers)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    model = CustomModel(cfg)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    preds = inference(cfg, model, test_dataloader)
    return preds


def get_model_path(cfg: "Config", fold: int) -> Path:
    return (
        cfg.output_dir / cfg.exp / "model" / f"{cfg.checkpoint_name.format(fold)}.ckpt"
    )


def setup(cfg: "Config") -> None:
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    (cfg.output_dir / cfg.exp).mkdir(exist_ok=True, parents=True)
    (cfg.output_dir / cfg.exp / "oof").mkdir(exist_ok=True, parents=True)
    (cfg.output_dir / cfg.exp / "submission").mkdir(exist_ok=True, parents=True)
    (cfg.output_dir / cfg.exp / "model").mkdir(exist_ok=True, parents=True)
    (cfg.output_dir / cfg.exp / "log").mkdir(exist_ok=True, parents=True)

    if cfg.is_wandb:
        wandb.login()


def metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def main():
    cfg = Config.set_config_from_argparse()

    setup(cfg)

    train_df = pd.read_csv(cfg.input_dir / "train_features.csv")
    test_df = pd.read_csv(cfg.input_dir / "test_features.csv")

    test_ids = test_df["ID"].values

    train_df = preprocess_features(train_df)
    test_df = preprocess_features(test_df)

    train_df = get_fold(cfg, train_df)

    oof = np.zeros((len(train_df), len(TARGET_COLS)))
    total_preds = list()
    for fold in range(cfg.n_folds):
        # Training
        train_fn(cfg, train_df, fold)

        # Inference
        model_path = get_model_path(cfg, fold)
        oof[train_df["fold"] == fold] = predict_fn(cfg, train_df, model_path)
        preds = predict_fn(cfg, test_df, model_path)

        preds = pd.DataFrame(preds, columns=TARGET_COLS)
        preds["ID"] = test_ids
        total_preds.append(preds)

    oof_df = pd.DataFrame(oof, columns=TARGET_COLS)
    oof_df["ID"] = train_df["ID"].values
    oof_df.to_csv(cfg.output_dir / cfg.exp / "oof" / "oof.csv", index=False)

    for target in TARGET_COLS:
        cv_score = metric(train_df[target].values, oof_df[target].values)
        print(f"cv {target} score: {cv_score:.4f}")
    total_cv_score = metric(train_df[TARGET_COLS].values, oof)
    print(f"cv total score: {total_cv_score:.4f}")

    total_preds = pd.concat(total_preds).mean(axis=0)
    total_preds.to_csv(
        cfg.output_dir
        / cfg.exp
        / "submission"
        / f"{cfg.exp}_{cfg.model_name}_{total_cv_score:.4f}_submission.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
