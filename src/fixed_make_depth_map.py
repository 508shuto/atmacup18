import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from glob import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ==================================
# 設定とパスのセットアップ
# ==================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PATH_TO_DATASET = "./input"


def get_relative_path(path):
    return os.path.join(PATH_TO_DATASET, path)


# 画像と特徴量へのパス
image_path_root_list = [
    get_relative_path("images/{ID}/image_t.png"),
    get_relative_path("images/{ID}/image_t-0.5.png"),
    get_relative_path("images/{ID}/image_t-1.0.png"),
]

train_feature_path = get_relative_path("train_features.csv")
test_feature_path = get_relative_path("test_features.csv")

# ==================================
# データフレームの読み込みと画像パスの追加
# ==================================
df_feature_train = pd.read_csv(train_feature_path)
df_feature_test = pd.read_csv(test_feature_path)

# データフレームに画像パスを追加
for t, key in enumerate(["img_path_t_00", "img_path_t_05", "img_path_t_10"]):
    df_feature_train[key] = [
        image_path_root_list[t].format(ID=ID) for ID in df_feature_train.ID
    ]
    df_feature_test[key] = [
        image_path_root_list[t].format(ID=ID) for ID in df_feature_test.ID
    ]

df_feature = pd.concat([df_feature_train, df_feature_test], axis=0, ignore_index=True)


# ==================================
# データセットの定義
# ==================================
class DepthDataset(Dataset):
    def __init__(self, dataframe, image_paths):
        self.dataframe = dataframe
        self.image_paths = image_paths

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        images = {}
        for t, path_template in enumerate(self.image_paths):
            img_path = path_template.format(ID=row.ID)
            img_pil = Image.open(img_path).convert("RGB")
            images[f"img_t_{t}"] = img_pil
        return row.ID, images


# ==================================
# 深度マップの生成関数
# ==================================
def generate_depth_maps(batch, model, processor):
    for ID, images in batch:
        for t, (key, img_pil) in enumerate(images.items()):
            # 深度推定のための画像準備
            inputs = processor(images=img_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            # 深度マップを元の画像サイズにリサイズ
            prediction = (
                torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=img_pil.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            # 深度マップの保存
            depth_path = (
                image_path_root_list[t].format(ID=ID).replace("images", "depth")
            )
            os.makedirs(os.path.dirname(depth_path), exist_ok=True)
            depth_map_image = (prediction / prediction.max() * 255).astype(np.uint8)
            Image.fromarray(depth_map_image).save(depth_path)


# ==================================
# 深度推定モデルとプロセッサの初期化
# ==================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
)
model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
).to(device)

# ==================================
# DataLoaderと処理の実行
# ==================================
dataset = DepthDataset(df_feature, image_path_root_list)
dataloader = DataLoader(
    dataset, batch_size=128, shuffle=False, num_workers=8, collate_fn=lambda x: x
)

for batch in tqdm(dataloader, total=len(dataloader)):
    generate_depth_maps(batch, model, processor)
