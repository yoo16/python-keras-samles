import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- モデルの読み込み ---
model = load_model("models/animal_classifier.h5")
print("Model loaded successfully.")

# --- クラス名の定義 ---
class_names = ["dog", "cat", "bird"]

# --- 画像の予測関数 ---
def predict_image(model, image_path):
    """画像をモデルに入力し、予測結果と確率を返す"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: '{image_path}' could not be loaded.")
        return None, None

    # 画像を64x64にリサイズし、正規化
    img_resized = cv2.resize(img, (64, 64)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)  # バッチ次元を追加

    # 予測の実行
    probabilities = model.predict(img_array)[0]
    prediction = np.argmax(probabilities)
    return class_names[prediction], probabilities[prediction]

# --- 検証用関数 ---
def test_images_in_folder(folder):
    """フォルダ内の画像を繰り返しテストし、結果を表示"""
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # 画像ファイルのみ対象
            file_path = os.path.join(folder, filename)
            prediction, probability = predict_image(model, file_path)
            if prediction is not None:
                print(f"Image: {filename} --> Prediction: {prediction} ({probability * 100:.2f}%)")
                visualize_prediction(file_path, prediction, probability)  # 予測結果の可視化

# --- 画像の可視化 ---
def visualize_prediction(image_path, prediction, probability):
    """画像に予測結果を描画し表示する"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load '{image_path}'")
        return

    # 予測結果と確率を描画
    label = f"{prediction}: {probability * 100:.2f}%"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 画像を表示し、ESCキーで終了
    cv2.imshow(f"{os.path.basename(image_path)}", img)
    key = cv2.waitKey(0) & 0xFF  # キー入力を待機
    if key == 27:  # ESCキーのASCIIコードは27
        print("ESC pressed. Exiting...")
        cv2.destroyAllWindows()
        exit(0)

    cv2.destroyAllWindows()

# --- 検証の実行 ---
test_folder = "data/test"  # テスト用画像フォルダのパス
test_images_in_folder(test_folder)
