import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

# --- 1. CIFAR-100 データセットの読み込み ---
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

# CIFAR-100の全クラス名
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

# --- 2. 使うクラス（dog, cat, bird）のインデックス ---
target_classes = ['dog', 'cat', 'bird']
target_indices = [class_names.index(cls) for cls in target_classes]

# --- 3. データセットのフィルタリング ---
def filter_data(x, y):
    mask = np.isin(y, target_indices)  # 指定したクラスのみ抽出
    x_filtered = x[mask.flatten()]
    y_filtered = y[mask]
    y_filtered = np.array([target_indices.index(label) for label in y_filtered])  # クラスIDを再マッピング
    return x_filtered, y_filtered

x_train_filtered, y_train_filtered = filter_data(x_train, y_train)
x_test_filtered, y_test_filtered = filter_data(x_test, y_test)

# --- 4. データの正規化 ---
x_train_filtered, x_test_filtered = x_train_filtered / 255.0, x_test_filtered / 255.0

# --- 5. モデルの構築 ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # 過学習を防ぐためのDropout
    layers.Dense(3, activation='softmax')  # 3クラスに対応
])

# モデルの概要を表示
model.summary()

# --- 6. モデルのコンパイル ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 7. EarlyStopping と ReduceLROnPlateau の設定 ---
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# --- 8. モデルの学習 ---
history = model.fit(
    x_train_filtered, y_train_filtered,
    epochs=50,
    batch_size=64,
    validation_data=(x_test_filtered, y_test_filtered),
    callbacks=[early_stopping, reduce_lr]
)

# --- 9. テストデータでモデルの評価 ---
test_loss, test_accuracy = model.evaluate(x_test_filtered, y_test_filtered, verbose=2)
print(f"\nTest accuracy: {test_accuracy}")

# --- 10. モデルの保存 ---
model.save('models/dog_cat_bird_model.h5')  # HDF5形式で保存
