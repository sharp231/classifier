from tensorflow import keras

# データセットの読み込み
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# データの形状を確認
print(f"訓練データ: {x_train.shape}")  # (60000, 28, 28)
print(f"訓練ラベル: {y_train.shape}")  # (60000,)
print(f"テストデータ: {x_test.shape}")  # (10000, 28, 28)
print(f"テストラベル: {y_test.shape}")  # (10000,)

# データの正規化（0-255 → 0-1）
x_train = x_train / 255.0
x_test = x_test / 255.0