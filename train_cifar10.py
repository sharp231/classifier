import tensorflow as tf
from tensorflow import keras

labels = ["飛行機","自動車","鳥","猫","鹿","犬","カエル","馬","船","トラック"]

# CIFAR-10読み込み（32x32 RGB, 10クラス）
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.reshape(-1)
y_test  = y_test.reshape(-1)

# 0-255 -> 0-1
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# シンプルなCNN（まず土台を作る）
model = keras.Sequential([
    keras.layers.Input(shape=(32, 32, 3)),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 学習（まず10epochでOK）
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=64,
)

# 保存（Flaskが読むファイル名に合わせる）
model.save("image_classifier.h5")
print("Saved: image_classifier.h5")