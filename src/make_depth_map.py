import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

# ==================================
# Dataの前処理に関するConfig
# ==================================
PATH_TO_DATASET = "./input"


def get_relative_path(path):
    return os.path.join(PATH_TO_DATASET, path)


# 画像へのパス
image_path_root_list = [
    get_relative_path("images/{ID}/image_t.png"),
    get_relative_path("images/{ID}/image_t-0.5.png"),
    get_relative_path("images/{ID}/image_t-1.0.png"),
]

# 特徴量のパス
train_feature_path = get_relative_path("train_features.csv")
traffic_light_path = get_relative_path("traffic_lights/{ID}.json")

# 信号機の情報へのパス
test_feature_path = get_relative_path("test_features.csv")


# ========================================
# DataFrameの読み込み
# ========================================
df_feature_train = pd.read_csv(train_feature_path)
df_feature_test = pd.read_csv(test_feature_path)

# =======================================
# 画像のパスの追加
# =======================================
df_feature_train["img_path_t_00"] = [
    image_path_root_list[0].format(ID=ID) for ID in df_feature_train.ID
]
df_feature_train["img_path_t_05"] = [
    image_path_root_list[1].format(ID=ID) for ID in df_feature_train.ID
]
df_feature_train["img_path_t_10"] = [
    image_path_root_list[2].format(ID=ID) for ID in df_feature_train.ID
]

df_feature_test["img_path_t_00"] = [
    image_path_root_list[0].format(ID=ID) for ID in df_feature_test.ID
]
df_feature_test["img_path_t_05"] = [
    image_path_root_list[1].format(ID=ID) for ID in df_feature_test.ID
]
df_feature_test["img_path_t_10"] = [
    image_path_root_list[2].format(ID=ID) for ID in df_feature_test.ID
]

df_feature = pd.concat([df_feature_train, df_feature_test], axis=0, ignore_index=True)

# =======================================
# Depth Mapの生成と保存
# =======================================
depth_anything_v2 = pipeline(
    task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
)

for i in tqdm(range(len(df_feature))):
    row = df_feature.iloc[i]

    for t, image_path_root in enumerate(image_path_root_list):
        img_pil = Image.open(image_path_root.format(ID=row.ID))
        pred = depth_anything_v2(img_pil)

        depth_path = image_path_root.format(ID=row.ID).replace("images", "depth")
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        pred["depth"].save(depth_path)
