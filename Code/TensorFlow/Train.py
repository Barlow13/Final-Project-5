"""
Road Object Detection Training Script
Optimized for RP2040 Deployment with TensorFlow Lite

This script trains a lightweight MobileNetV2-based model to detect common road objects
and exports an optimized TFLite model suitable for the SparkFun Thing Plus RP2040.
"""

import os
import random
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow_datasets as tfds
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# ─── SUPPRESS LOGGING ─────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
BATCH_SIZE = 32
FROZEN_EPOCHS = 10
FINE_TUNE_EPOCHS = 30
INITIAL_LR = 1e-3
IMG_HEIGHT, IMG_WIDTH = 64, 64
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
SEED = 42
NICKNAME = 'RoadLiteMobileNetV2'
EXPORT_DIR = './export'
DATA_DIR = './Data'
EXCEL_DIR = './excel'
PREDICTION_DIR = './predictions'

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Road object classes to detect
ROAD_CLASSES = ['bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']
NUM_CLASSES = len(ROAD_CLASSES)

# Create necessary directories
for directory in [EXPORT_DIR, DATA_DIR, EXCEL_DIR, PREDICTION_DIR]:
    os.makedirs(directory, exist_ok = True)

# ─── DATA PREPARATION ─────────────────────────────────────────────────────────
TRAIN_EXCEL_PATH = os.path.join(EXCEL_DIR, 'train.xlsx')
TEST_EXCEL_PATH = os.path.join(EXCEL_DIR, 'test.xlsx')

# Load or create dataset
if not os.path.exists(TRAIN_EXCEL_PATH):
    print("Creating dataset from COCO...")
    dataset, info = tfds.load('coco/2017', split = ['train', 'validation'], shuffle_files = True, with_info = True)
    label_names = info.features['objects']['label'].names
    road_label_ids = [label_names.index(name) for name in ROAD_CLASSES]


    def process_split(split, split_name, max_samples = 5000):
        """Extract road objects from COCO dataset and save as images"""
        records = []
        for i, ex in enumerate(tfds.as_numpy(split)):
            if i >= max_samples:
                break

            labels = set(ex['objects']['label'].tolist())
            target = [1 if lid in labels else 0 for lid in road_label_ids]

            # Skip samples with no target objects to balance dataset
            if sum(target) == 0 and random.random() > 0.2:
                continue

            fname = f"{split_name}_{i:06d}.jpg"
            path = os.path.join(DATA_DIR, fname)
            img = cv2.resize(ex['image'], IMG_SIZE)
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            records.append({
                'filename':fname,
                'target_bin':','.join(map(str, target)),
                'target':','.join([ROAD_CLASSES[j] for j, v in enumerate(target) if v]) or 'None'
            })

        return pd.DataFrame(records)


    df_train = process_split(dataset[0], 'train')
    df_test = process_split(dataset[1], 'test')

    # Save datasets
    df_train.to_excel(TRAIN_EXCEL_PATH, index = False)
    df_test.to_excel(TEST_EXCEL_PATH, index = False)

    print(f"Created dataset with {len(df_train)} training and {len(df_test)} testing samples")
else:
    print("Loading existing dataset...")
    df_train = pd.read_excel(TRAIN_EXCEL_PATH)
    df_test = pd.read_excel(TEST_EXCEL_PATH)
    print(f"Loaded dataset with {len(df_train)} training and {len(df_test)} testing samples")


# ─── IMAGE AUGMENTATION FUNCTIONS ───────────────────────────────────────────────
def decode_image(path):
    """Load and normalize an image from path"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, IMG_SIZE)
    return img / 255.0


def augment_image(image):
    """Apply various augmentations to the input image"""
    # Random brightness and contrast
    image = tf.image.random_brightness(image, max_delta = 0.2)
    image = tf.image.random_contrast(image, lower = 0.8, upper = 1.2)

    # Random flip and rotation
    image = tf.image.random_flip_left_right(image)
    k = tf.random.uniform(shape = [], minval = 0, maxval = 4, dtype = tf.int32)
    image = tf.image.rot90(image, k = k)

    # Random hue/saturation
    image = tf.image.random_hue(image, max_delta = 0.1)
    image = tf.image.random_saturation(image, lower = 0.8, upper = 1.2)

    # Ensure values stay in [0, 1] range
    return tf.clip_by_value(image, 0, 1)


def mixup(images, labels, alpha = 0.2):
    """Apply mixup augmentation to a batch of images and labels"""
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    lam = tf.random.uniform(shape = [], minval = 0, maxval = alpha)

    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)

    return mixed_images, mixed_labels


# ─── DATASET CREATION ───────────────────────────────────────────────────────
def build_dataset(df, augment = False, shuffle = True, apply_mixup = False):
    """Build a TensorFlow dataset from DataFrame"""
    paths = [os.path.join(DATA_DIR, f) for f in df['filename']]
    labels = [list(map(int, r.split(','))) for r in df['target_bin']]

    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Decode images and convert labels to float
    ds = ds.map(
        lambda x, y:(decode_image(x), tf.cast(y, tf.float32)),
        num_parallel_calls = tf.data.AUTOTUNE
    )

    # Apply augmentation if requested
    if augment:
        ds = ds.map(
            lambda x, y:(augment_image(x), y),
            num_parallel_calls = tf.data.AUTOTUNE
        )

    # Shuffle if requested
    if shuffle:
        ds = ds.shuffle(1024)

    # Batch dataset
    ds = ds.batch(BATCH_SIZE)

    # Apply mixup if requested
    if apply_mixup:
        ds = ds.map(
            lambda x, y:mixup(x, y, alpha = 0.2),
            num_parallel_calls = tf.data.AUTOTUNE
        )

    # Prefetch for performance
    return ds.prefetch(tf.data.AUTOTUNE)


# Create datasets
train_ds = build_dataset(df_train, augment = True, shuffle = True, apply_mixup = True)
test_ds = build_dataset(df_test, shuffle = False)


# ─── MODEL BUILDING ──────────────────────────────────────────────────────────
def build_model(trainable_base = False, fine_tuning = False):
    """Build the MobileNetV2-based model"""
    # Load MobileNetV2 with pre-trained weights, using alpha=0.35 for a smaller model
    base_model = MobileNetV2(
        input_shape = (*IMG_SIZE, 3),
        include_top = False,
        weights = 'imagenet',
        alpha = 0.35  # Lighter model
    )

    # Set base model trainable status
    base_model.trainable = trainable_base

    # If fine-tuning, only make the last few layers trainable
    if fine_tuning:
        for layer in base_model.layers[:-20]:
            layer.trainable = False

    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(128, activation = 'relu'),
        layers.Dropout(0.3),  # Additional dropout
        layers.Dense(NUM_CLASSES, activation = 'sigmoid')
    ])

    # Compile with appropriate learning rate
    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            INITIAL_LR if not fine_tuning else INITIAL_LR * 0.1,
            clipnorm = 1.0  # Gradient clipping for stability
        ),
        loss = 'binary_crossentropy',
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name = 'binary_accuracy'),
            tf.keras.metrics.Precision(name = 'precision'),
            tf.keras.metrics.Recall(name = 'recall'),
            tf.keras.metrics.AUC(name = 'auc')
        ]
    )

    return model


# ─── VISUALIZATION UTILITIES ───────────────────────────────────────────────────
def visualize_predictions(model, df, classes, thresholds = None, num_samples = 8):
    """Visualize model predictions on random samples"""
    if thresholds is None:
        thresholds = [0.5] * len(classes)

    samples = df.sample(num_samples)
    imgs = []
    trues = []

    # Prepare images and true labels
    for _, row in samples.iterrows():
        img_path = os.path.join(DATA_DIR, row['filename'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        imgs.append(img / 255.0)
        trues.append(list(map(int, row['target_bin'].split(','))))

    # Convert to arrays
    imgs = np.array(imgs)
    trues = np.array(trues)

    # Get predictions
    preds = model.predict(imgs, verbose = 0)
    preds_bin = (preds > thresholds).astype(int)

    # Create visualization grid
    fig, axes = plt.subplots(2, 4, figsize = (16, 8))
    for i, ax in enumerate(axes.flatten()):
        img = (imgs[i] * 255).astype(np.uint8)
        true_labels = [classes[j] for j, v in enumerate(trues[i]) if v]
        pred_labels = [classes[j] for j, v in enumerate(preds_bin[i]) if v]

        # Top 3 predictions with probabilities
        top_indices = np.argsort(preds[i])[::-1][:3]
        top_preds = [(classes[j], preds[i][j]) for j in top_indices]

        # Display image
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(
            f"True: {', '.join(true_labels) or 'None'}\n"
            f"Pred: {', '.join(pred_labels) or 'None'}\n"
            f"Top-3: {', '.join([f'{k}({v:.2f})' for k, v in top_preds])}",
            fontsize = 8
        )

    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_DIR, f"prediction_grid.png"))
    plt.close()
    return preds, trues


# ─── TRAINING CALLBACK ──────────────────────────────────────────────────────
class MetricsCallback(tf.keras.callbacks.Callback):
    """Custom callback to track F1, precision, and recall metrics"""

    def __init__(self, test_ds, classes, df_test):
        super().__init__()
        self.test_ds = test_ds
        self.classes = classes
        self.df_test = df_test
        self.metrics_history = {
            'f1':[], 'precision':[], 'recall':[]
        }
        self.thresholds = np.array([0.5] * len(classes))

    def on_epoch_end(self, epoch, logs = None):
        # Get predictions and true labels
        y_true, y_pred = [], []
        for x, y in self.test_ds:
            y_true.append(y.numpy())
            y_pred.append(self.model.predict(x, verbose = 0))

        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)

        # Optimize thresholds for F1 score (simplified approach)
        if (epoch + 1) % 5 == 0:
            # Only optimize every 5 epochs to save time
            for i in range(NUM_CLASSES):
                best_f1 = 0
                best_thresh = 0.5

                for t in np.linspace(0.3, 0.7, 5):
                    preds_bin = (y_pred[:, i] > t).astype(int)
                    f1 = f1_score(y_true[:, i], preds_bin, zero_division = 0)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = t

                self.thresholds[i] = best_thresh

        # Apply thresholds
        y_pred_bin = (y_pred > self.thresholds).astype(int)

        # Calculate metrics
        f1 = f1_score(y_true, y_pred_bin, average = 'macro', zero_division = 0)
        precision = precision_score(y_true, y_pred_bin, average = 'macro', zero_division = 0)
        recall = recall_score(y_true, y_pred_bin, average = 'macro', zero_division = 0)

        # Store metrics
        self.metrics_history['f1'].append(f1)
        self.metrics_history['precision'].append(precision)
        self.metrics_history['recall'].append(recall)

        # Add to logs for other callbacks
        logs = logs or {}
        logs['val_f1_macro'] = f1

        # Print progress
        print(f"Epoch {epoch + 1}: "
              f"F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, "
              f"Thresholds: {np.mean(self.thresholds):.2f}")

        # Visualize predictions every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            visualize_predictions(self.model, self.df_test, self.classes, self.thresholds)

    def on_train_end(self, logs = None):
        # Plot metrics history
        plt.figure(figsize = (10, 6))
        for metric_name, values in self.metrics_history.items():
            plt.plot(range(1, len(values) + 1), values, marker = 'o', label = metric_name)

        plt.title('Model Metrics Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(EXPORT_DIR, 'metrics_history.png'))
        plt.close()

        # Save thresholds
        np.save(os.path.join(EXPORT_DIR, 'best_thresholds.npy'), self.thresholds)
        print(f"Saved optimized thresholds to {os.path.join(EXPORT_DIR, 'best_thresholds.npy')}")

        # Generate final visualization and report
        y_pred, y_true = visualize_predictions(self.model, self.df_test, self.classes, self.thresholds)
        y_pred_bin = (y_pred > self.thresholds).astype(int)

        # Save classification report
        report = classification_report(y_true, y_pred_bin, target_names = self.classes)
        with open(os.path.join(EXPORT_DIR, 'classification_report.txt'), 'w') as f:
            f.write(report)

        print("\nFinal Classification Report:")
        print(report)


# ─── STAGE 1: FROZEN BASE TRAINING ───────────────────────────────────────────
print("\n=== Stage 1: Training with Frozen Base Model ===")
stage1_model = build_model(trainable_base = False)

# Callbacks
metrics_callback = MetricsCallback(test_ds, ROAD_CLASSES, df_test)
early_stop = EarlyStopping(
    monitor = 'val_f1_macro',
    mode = 'max',
    patience = 7,
    restore_best_weights = True,
    verbose = 1
)
checkpoint = ModelCheckpoint(
    os.path.join(EXPORT_DIR, 'best_model_stage1.keras'),
    monitor = 'val_f1_macro',
    mode = 'max',
    save_best_only = True,
    verbose = 1
)
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_f1_macro',
    mode = 'max',
    factor = 0.5,
    patience = 3,
    min_lr = 1e-6,
    verbose = 1
)

# Train stage 1
history_stage1 = stage1_model.fit(
    train_ds,
    epochs = FROZEN_EPOCHS,
    validation_data = test_ds,
    callbacks = [metrics_callback, early_stop, checkpoint, reduce_lr],
    verbose = 1
)

# ─── STAGE 2: FINE TUNING ───────────────────────────────────────────────────
print("\n=== Stage 2: Fine-tuning Model ===")
# Load best stage 1 model
stage1_model = tf.keras.models.load_model(
    os.path.join(EXPORT_DIR, 'best_model_stage1.keras')
)

# Create fine-tuning model
stage2_model = build_model(trainable_base = True, fine_tuning = True)

# Copy weights from stage 1
stage2_model.set_weights(stage1_model.get_weights())

# Callbacks for stage 2
metrics_callback_stage2 = MetricsCallback(test_ds, ROAD_CLASSES, df_test)
early_stop_stage2 = EarlyStopping(
    monitor = 'val_f1_macro',
    mode = 'max',
    patience = 10,
    restore_best_weights = True,
    verbose = 1
)
checkpoint_stage2 = ModelCheckpoint(
    os.path.join(EXPORT_DIR, 'best_model_final.keras'),
    monitor = 'val_f1_macro',
    mode = 'max',
    save_best_only = True,
    verbose = 1
)
reduce_lr_stage2 = ReduceLROnPlateau(
    monitor = 'val_f1_macro',
    mode = 'max',
    factor = 0.2,
    patience = 5,
    min_lr = 1e-7,
    verbose = 1
)

# Train stage 2
history_stage2 = stage2_model.fit(
    train_ds,
    epochs = FINE_TUNE_EPOCHS,
    validation_data = test_ds,
    callbacks = [metrics_callback_stage2, early_stop_stage2, checkpoint_stage2, reduce_lr_stage2],
    verbose = 1
)

# ─── EXPORT TO TFLITE ────────────────────────────────────────────────────────
print("\n=== Converting to TFLite for RP2040 Deployment ===")

# Load the best model
final_model = tf.keras.models.load_model(
    os.path.join(EXPORT_DIR, 'best_model_final.keras')
)

# Summary of the model
final_model.summary()


# Define a representative dataset for quantization
def representative_dataset():
    """Generate representative dataset for quantization"""
    for i in range(100):
        row = df_train.sample(1).iloc[0]
        img_path = os.path.join(DATA_DIR, row['filename'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        yield [np.expand_dims(img, axis = 0)]


# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)

# Apply optimizations for RP2040
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.uint8

# Convert model
tflite_model = converter.convert()

# Save TFLite model
tflite_path = os.path.join(EXPORT_DIR, f"{NICKNAME}.tflite")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved: {tflite_path}")
print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")

# Also save a metadata file with class names for inference
class_names = {i:name for i, name in enumerate(ROAD_CLASSES)}
with open(os.path.join(EXPORT_DIR, "class_names.txt"), 'w') as f:
    for i, name in class_names.items():
        f.write(f"{i}: {name}\n")

print("Training complete! The optimized TFLite model is ready for deployment on the SparkFun Thing Plus RP2040.")