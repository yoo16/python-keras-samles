import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- モデルの読み込み ---
model = load_model("models/animal_classifier.h5")
print("Model loaded successfully.")

# --- クラス名の定義 ---
class_names = ["dog", "cat", "crow"]

# --- 画像の予測関数 ---
def predict_frame(model, frame):
    frame_resized = cv2.resize(frame, (64, 64)) / 255.0
    frame_array = np.expand_dims(frame_resized, axis=0)  # バッチ次元を追加

    # 予測の実行
    probabilities = model.predict(frame_array)[0]
    prediction = np.argmax(probabilities)
    return class_names[prediction], probabilities[prediction]

# --- 動画からカラスを検出する関数 ---
def detect_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return

    paused = False  # 一時停止フラグ

    while cap.isOpened():
        if not paused:  # 一時停止中でなければフレームを取得
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break

            # 予測の実行
            prediction, probability = predict_frame(model, frame)

            # 結果を表示
            visualize_frame(frame, prediction, probability)

        # キー入力の待機（1ms）
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESCキーで終了
            print("ESC pressed. Exiting...")
            break
        elif key == 32:  # スペースキーで一時停止/再開
            paused = not paused  # 一時停止状態を反転

    cap.release()
    cv2.destroyAllWindows()

# --- 予測結果をフレームに描画して表示 ---
def visualize_frame(frame, prediction, probability):
    if probability >= 0.5:
        label = f"{prediction}: {probability * 100:.2f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # フレームを表示
    cv2.imshow("Detection", frame)

# --- メイン処理 ---
if __name__ == "__main__":
    video_name = input("Enter the video file name (without .mp4): ")
    video_path = f"videos/{video_name}.mp4"

    if not os.path.exists(video_path):
        print(f"Error: The file '{video_path}' does not exist.")
    else:
        # 動画検証の実行
        detect_in_video(video_path)
