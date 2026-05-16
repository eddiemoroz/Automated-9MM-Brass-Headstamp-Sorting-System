import os
import json
import numpy as np
import tensorflow as tf

datasetDir =
imgSize = (320, 320)
batchSize = 16
epochs = 15
seed = 42

modelName = "headstampEfficientnetWithUnknown"

unknownLabel = "UNKNOWN"

# Reduces UNKNOWN influence on model as a whole
unknownClassWeight = 0.25

trainData = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(datasetDir, "train"),
    image_size=imgSize,
    batch_size=batchSize,
    label_mode="int",
    color_mode="rgb",
    shuffle=True,
    seed=seed
)

valData = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(datasetDir, "val"),
    image_size=imgSize,
    batch_size=batchSize,
    label_mode="int",
    color_mode="rgb",
    shuffle=False
)

classNames = trainData.class_names
numClasses = len(classNames)

print("\nClass Names:", classNames)
print("Num Classes:", numClasses)

unknownIndex = classNames.index(unknownLabel)
print("UNKNOWN index:", unknownIndex)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomContrast(0.25),
], name="augmentation")

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_train(images, labels):
    images = tf.cast(images, tf.float32)
    images = data_augmentation(images, training=True)
    return images, labels

def preprocess_val(images, labels):
    images = tf.cast(images, tf.float32)
    return images, labels

train_data = trainData.map(preprocess_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_data = valData.map(preprocess_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

base_model = tf.keras.applications.EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=imgSize + (3,)
)

base_model.trainable = False

inputs = tf.keras.layers.Input(shape=imgSize + (3,))
x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(numClasses, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

class_weights = {i: 1.0 for i in range(numClasses)}
class_weights[unknownIndex] = unknownClassWeight

print("\nClass weights:")
for i, name in enumerate(classNames):
    print(f"{i:2d} {name:15s} weight={class_weights[i]}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.summary()

model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    class_weight=class_weights,
)

print("\nFine Tuning:")

base_model.trainable = True

# Freeze most layers, unfreeze last 40
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    class_weight=class_weights,
)

model.save(f"{modelName}.keras")

with open(f"{modelName}_classes.json", "w") as f:
    json.dump(modelName, f)

print("\nSaved model to:", f"{modelName}.keras")
print("Saved class names to:", f"{modelName}_classes.json")
