import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
import os

# -----------------------------
# 1️⃣ Paths & parameters
# -----------------------------
DATASET_DIR = "Medicinal_Leaf_Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 20
EPOCHS_FINE = 15
LEARNING_RATE = 1e-4
SEED = 42

# -----------------------------
# 2️⃣ Data Generators (IMPROVED)
# -----------------------------

# 🔥 Strong augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

# 🚫 No augmentation for validation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
print(f"✅ Number of classes: {NUM_CLASSES}")

# -----------------------------
# 3️⃣ Save class indices
# -----------------------------
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f, indent=4)

print("✅ class_indices.json saved")

# -----------------------------
# 4️⃣ Load MobileNet
# -----------------------------
base_model = MobileNet(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# -----------------------------
# 5️⃣ Custom Head
# -----------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# -----------------------------
# 6️⃣ Compile (Head Training)
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 7️⃣ Callbacks (VERY IMPORTANT)
# -----------------------------
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True),
    ReduceLROnPlateau(factor=0.3, patience=3)
]

# -----------------------------
# 8️⃣ Train Top Layers
# -----------------------------
print("🚀 Training classifier head...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks
)

# -----------------------------
# 9️⃣ Fine-Tuning
# -----------------------------
print("🔧 Fine-tuning MobileNet...")

base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-25]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE,
    callbacks=callbacks
)

# -----------------------------
# 🔟 Save Final Model
# -----------------------------
model.save("final_model.keras")
print("✅ final_model.keras saved successfully")
