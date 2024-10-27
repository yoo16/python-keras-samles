import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 画像の読み込み関数 ---
def load_images_from_folder(base_folder):
    images = []
    labels = []
    label_dict = {"dogs": 0, "cats": 1, "birds": 2}  # クラスとラベルの対応

    for label_name, label in label_dict.items():
        class_folder = os.path.join(base_folder, label_name)
        if not os.path.exists(class_folder):
            print(f"Warning: Folder '{class_folder}' does not exist.")
            continue

        for filename in os.listdir(class_folder):
            file_path = os.path.join(class_folder, filename)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # カラー画像で読み込み
            if img is not None:
                resized = cv2.resize(img, (64, 64))  # サイズを64x64に変更
                images.append(resized)
                labels.append(label)

    return np.array(images), np.array(labels)

# --- データの読み込み ---
base_folder = "data"
X, y = load_images_from_folder(base_folder)

# --- データの前処理 ---
X = X / 255.0  # ピクセル値を正規化
y = to_categorical(y, 3)  # ラベルをOne-Hotエンコーディング

# --- データの分割 (80% 訓練データ, 20% テストデータ) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- データ拡張の設定 ---
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.3,
    shear_range=0.2,
    brightness_range=[0.8, 1.2]
)
datagen.fit(X_train)

# --- VGG16ベースのモデルを構築 ---
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# 上位層に全結合層を追加
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# VGG16の重みを固定
for layer in base_model.layers:
    layer.trainable = False

# 最後の層のみ訓練する
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# --- モデルの学習 ---
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=20, validation_data=(X_test, y_test))

# --- モデルの保存 ---
model.save("models/animal_classifier.h5")
print("Model saved as 'animal_classifier.h5'")

# --- テストデータで評価 ---
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# --- 学習の進捗を視覚化 ---
def plot_training_history(history):
    """学習の進捗（精度と損失）をプロット"""
    # 精度のプロット
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 損失のプロット
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # グラフを表示
    plt.show()

# --- 学習履歴の可視化を実行 ---
plot_training_history(history)