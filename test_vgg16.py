from tensorflow.keras.applications import VGG16

# VGG16モデルのロード（事前学習済み）
model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# モデルの概要を表示
model.summary()