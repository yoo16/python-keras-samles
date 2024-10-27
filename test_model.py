import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os

# モデルのパスを指定
model_path = 'models/cifar100_model.h5'
loaded_model = tf.keras.models.load_model(model_path)

# 動物クラスのリスト
animal_classes = [
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm'
]

# CIFAR-100 全クラス名（100クラス）
class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# 画像を読み込んで前処理する関数
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(32, 32))  # CIFAR-100のサイズにリサイズ
    img_array = img_to_array(img) / 255.0  # 正規化
    return np.expand_dims(img_array, axis=0)  # バッチ次元を追加

# フォルダ内のすべての画像を検証する関数
def predict_images_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 画像ファイルを対象
            img_path = os.path.join(folder_path, filename)
            img_array = load_and_preprocess_image(img_path)

            # モデルで予測
            predictions = loaded_model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]

            # 動物クラスかどうかを判別
            if predicted_class in animal_classes:
                label = f"Animal: {predicted_class}"
            else:
                label = "Non-animal"

            # 結果を表示
            print(f"Image: {filename} -> {label}")

            # 画像と予測結果をプロット
            plt.imshow(load_img(img_path))
            plt.title(label)
            plt.axis('off')
            plt.show()

# data/test フォルダ内の画像を検証
predict_images_from_folder('data/test')
