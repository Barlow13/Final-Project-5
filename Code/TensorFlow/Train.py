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
FROZEN_EPOCHS = 16
FINE_TUNE_EPOCHS = 256
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
    os.makedirs(directory, exist_ok=True)

# ─── DATA PREPARATION ─────────────────────────────────────────────────────────
TRAIN_EXCEL_PATH = os.path.join(EXCEL_DIR, 'train.xlsx')
TEST_EXCEL_PATH = os.path.join(EXCEL_DIR, 'test.xlsx')

# Load or create dataset
if not os.path.exists(TRAIN_EXCEL_PATH):
    print("Creating dataset from COCO...")
    dataset, info = tfds.load('coco/2017', split=['train', 'validation'], shuffle_files=True, with_info=True)
    label_names = info.features['objects']['label'].names
    road_label_ids = [label_names.index(name) for name in ROAD_CLASSES]

    def process_split(split, split_name, max_samples=10000):
        """Extract road objects with better class balance"""
        records = []
        class_counters = {class_name: 0 for class_name in ROAD_CLASSES}
        target_per_class = max_samples // (2 * len(ROAD_CLASSES))  # Target samples per class

        for i, ex in enumerate(tfds.as_numpy(split)):
            if sum(class_counters.values()) >= max_samples:
                break

            labels = set(ex['objects']['label'].tolist())
            target = [1 if lid in labels else 0 for lid in road_label_ids]

            # Count the classes in this image
            classes_present = [ROAD_CLASSES[j] for j, v in enumerate(target) if v]

            # Skip if no target classes or if we already have enough of these classes
            if not classes_present:
                # Allow some "None" samples but limit them
                if random.random() > 0.2 or sum(class_counters.values()) > 0.7 * max_samples:
                    continue
            else:
                # Check if we already have enough samples of these classes
                if all(class_counters[cls] >= target_per_class for cls in classes_present):
                    if random.random() > 0.3:  # Still keep some with probability 0.3
                        continue

                # Update class counters
                for cls in classes_present:
                    class_counters[cls] += 1

            # Basic image quality filtering
            img = ex['image']
            if np.mean(img) < 20 or np.mean(img) > 235:  # Skip very dark/bright images
                continue

            # Skip images with extreme aspect ratios
            h, w = img.shape[0], img.shape[1]
            if max(h, w) / min(h, w) > 3:
                continue

            # Process and save image
            fname = f"{split_name}_{i:06d}.jpg"
            path = os.path.join(DATA_DIR, fname)
            img = cv2.resize(img, IMG_SIZE)
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            records.append({
                'filename': fname,
                'target_bin': ','.join(map(str, target)),
                'target': ','.join(classes_present) or 'None'
            })

            # Print progress periodically
            if len(records) % 500 == 0:
                print(f"Processed {len(records)} records. Class distribution: {class_counters}")

        print(f"Final class distribution: {class_counters}")
        return pd.DataFrame(records)

    df_train = process_split(dataset[0], 'train')
    df_test = process_split(dataset[1], 'test')

    # Save datasets
    df_train.to_excel(TRAIN_EXCEL_PATH, index=False)
    df_test.to_excel(TEST_EXCEL_PATH, index=False)

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
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img / 255.0

def augment_image(image):
    """Enhanced augmentation to improve model generalization"""
    # Standard augmentations
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)

    # Add random noise occasionally to improve robustness
    if tf.random.uniform(shape=[], minval=0, maxval=1) < 0.3:
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02)
        image = image + noise

    # More aggressive geometric transformations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random rotation (0, 90, 180, 270 degrees)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)

    # Random crop and resize
    if tf.random.uniform(shape=[], minval=0, maxval=1) < 0.5:
        scale = tf.random.uniform(shape=[], minval=0.8, maxval=1.0)
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
        offset_h = tf.random.uniform(shape=[], maxval=h - new_h + 1, dtype=tf.int32)
        offset_w = tf.random.uniform(shape=[], maxval=w - new_w + 1, dtype=tf.int32)
        image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, new_h, new_w)
        image = tf.image.resize(image, size=[h, w])

    # Ensure values stay in [0, 1] range
    return tf.clip_by_value(image, 0, 1)

def mixup(images, labels, alpha=0.2):
    """Apply mixup augmentation to a batch of images and labels"""
    batch_size = tf.shape(images)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    lam = tf.random.uniform(shape=[], minval=0, maxval=alpha)

    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)

    return mixed_images, mixed_labels

# ─── DATASET CREATION ───────────────────────────────────────────────────────
def build_dataset(df, augment=False, shuffle=True, apply_mixup=False):
    """Build a TensorFlow dataset from DataFrame"""
    paths = [os.path.join(DATA_DIR, f) for f in df['filename']]
    labels = [list(map(int, r.split(','))) for r in df['target_bin']]

    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Decode images and convert labels to float
    ds = ds.map(
        lambda x, y: (decode_image(x), tf.cast(y, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Apply augmentation if requested
    if augment:
        ds = ds.map(
            lambda x, y: (augment_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Shuffle if requested
    if shuffle:
        ds = ds.shuffle(1024)

    # Batch dataset
    ds = ds.batch(BATCH_SIZE)

    # Apply mixup if requested
    if apply_mixup:
        ds = ds.map(
            lambda x, y: mixup(x, y, alpha=0.2),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Prefetch for performance
    return ds.prefetch(tf.data.AUTOTUNE)

# Create datasets
train_ds = build_dataset(df_train, augment=True, shuffle=True, apply_mixup=True)
test_ds = build_dataset(df_test, shuffle=False)

# ─── DEFINE CUSTOM LOSS ────────────────────────────────────────────────────
# Properly implement focal loss as a subclass for better serialization
@tf.keras.utils.register_keras_serializable(package="custom_losses")
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss implementation to focus on hard-to-classify examples"""

    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Clip prediction values to avoid numerical instability
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Apply focal weighting
        loss = self.alpha * tf.math.pow(1 - y_pred, self.gamma) * cross_entropy * y_true + \
               (1 - self.alpha) * tf.math.pow(y_pred, self.gamma) * cross_entropy * (1 - y_true)

        # Return mean loss
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
        })
        return config

# ─── MODEL BUILDING ──────────────────────────────────────────────────────────
def build_model(trainable_base=False, fine_tuning=False):
    """Build the MobileNetV2-based model"""
    # Load MobileNetV2 with pre-trained weights, using alpha=0.35 for a smaller model
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35  # Lighter model
    )

    # Set base model trainable status
    base_model.trainable = trainable_base

    # If fine-tuning, only make the last few layers trainable
    if fine_tuning:
        for layer in base_model.layers[:-25]:
            layer.trainable = False

    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.6),  # Prevent overfitting
        layers.Dense(48, activation='relu'),
        layers.Dropout(0.4),  # Additional dropout
        layers.Dense(NUM_CLASSES, activation='sigmoid')
    ])

    # Use in model compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            INITIAL_LR if not fine_tuning else INITIAL_LR * 0.1,
            clipnorm=1.0
        ),
        loss=FocalLoss(),  # Using our properly serializable FocalLoss
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model

# ─── VISUALIZATION UTILITIES ───────────────────────────────────────────────────
def visualize_predictions(model, df, classes, thresholds=None, num_samples=8):
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
    preds = model.predict(imgs, verbose=0)
    preds_bin = (preds > thresholds).astype(int)

    # Create visualization grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
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
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_DIR, f"prediction_grid.png"))
    plt.close()
    return preds, trues

# ─── TRAINING CALLBACK ──────────────────────────────────────────────────────
class MetricsCallback(tf.keras.callbacks.Callback):

    def __init__(self, test_ds, classes, df_test):
        super().__init__()
        self.test_ds = test_ds
        self.classes = classes
        self.df_test = df_test
        self.metrics_history = {
            'f1': [], 'precision': [], 'recall': []
        }
        self.thresholds = np.array([0.5] * len(classes))
        # Class weights for threshold optimization
        self.class_weights = np.ones(len(classes))

    def on_epoch_end(self, epoch, logs=None):
        # Get predictions and true labels
        y_true, y_pred = [], []
        for x, y in self.test_ds:
            y_true.append(y.numpy())
            y_pred.append(self.model.predict(x, verbose=0))

        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)

        # Calculate class distribution to identify rare classes
        class_counts = np.sum(y_true, axis=0)
        total_positives = np.sum(class_counts)

        # Update class weights based on distribution
        if total_positives > 0:
            self.class_weights = np.ones(len(self.classes))
            for i, count in enumerate(class_counts):
                if count > 0:
                    # Inversely weight classes by their frequency
                    self.class_weights[i] = total_positives / (count * len(self.classes))
                    # Cap at 5.0 to prevent extreme values
                    self.class_weights[i] = min(5.0, self.class_weights[i])

        # Optimize thresholds every 5 epochs
        if (epoch + 1) % 5 == 0:
            for i, class_name in enumerate(self.classes):
                best_f1 = 0
                best_thresh = 0.5

                # Use more thresholds for rare classes, fewer for common ones
                if class_counts[i] < 5:
                    # For very rare classes, use more granular thresholds
                    thresholds = np.linspace(0.05, 0.7, 40)
                else:
                    # For common classes, use fewer threshold points
                    thresholds = np.linspace(0.1, 0.8, 20)

                for t in thresholds:
                    preds_bin = (y_pred[:, i] > t).astype(int)

                    # For rare classes, weight recall higher than precision
                    if class_counts[i] < 5:
                        precision = precision_score(y_true[:, i], preds_bin, zero_division=0)
                        recall = recall_score(y_true[:, i], preds_bin, zero_division=0)
                        # Use F2 score to emphasize recall for rare classes
                        if precision > 0 and recall > 0:
                            f_score = (5 * precision * recall) / (4 * precision + recall)
                        else:
                            f_score = 0
                    else:
                        # Use regular F1 for common classes
                        f_score = f1_score(y_true[:, i], preds_bin, zero_division=0)

                    if f_score > best_f1:
                        best_f1 = f_score
                        best_thresh = t

                self.thresholds[i] = best_thresh

        # Apply thresholds
        y_pred_bin = (y_pred > self.thresholds).astype(int)

        # Calculate metrics
        f1 = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred_bin, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred_bin, average='macro', zero_division=0)

        # Calculate class-specific F1 scores for tracking
        class_f1 = {}
        for i, class_name in enumerate(self.classes):
            class_f1[class_name] = f1_score(y_true[:, i], y_pred_bin[:, i], zero_division=0)

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
        print(f"Class F1 scores: {', '.join([f'{c}={v:.2f}' for c, v in class_f1.items()])}")

        # Visualize predictions every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            visualize_predictions(self.model, self.df_test, self.classes, self.thresholds)

    def on_train_end(self, logs=None):
        # Plot metrics history
        plt.figure(figsize=(10, 6))
        for metric_name, values in self.metrics_history.items():
            plt.plot(range(1, len(values) + 1), values, marker='o', label=metric_name)

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

        # Generate visual predictions on a small subset (for visualization only)
        visualize_predictions(self.model, self.df_test, self.classes, self.thresholds)

        # IMPORTANT: Evaluate on the FULL test dataset, not just a few samples
        # Get predictions for ALL test data
        y_true, y_pred = [], []
        for x, y in self.test_ds:
            y_true.append(y.numpy())
            y_pred.append(self.model.predict(x, verbose=0))

        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        y_pred_bin = (y_pred > self.thresholds).astype(int)

        # Save classification report based on full test set
        report = classification_report(y_true, y_pred_bin, target_names=self.classes)
        with open(os.path.join(EXPORT_DIR, 'classification_report.txt'), 'w') as f:
            f.write(report)

        print("\nFinal Classification Report (on full test dataset):")
        print(report)

# ─── STAGE 1: FROZEN BASE TRAINING ───────────────────────────────────────────
print("\n=== Stage 1: Training with Frozen Base Model ===")
stage1_model = build_model(trainable_base=False)

# Callbacks
metrics_callback = MetricsCallback(test_ds, ROAD_CLASSES, df_test)
early_stop = EarlyStopping(
    monitor='val_f1_macro',
    mode='max',
    patience=8,
    restore_best_weights=True,
    verbose=1
)
checkpoint = ModelCheckpoint(
    os.path.join(EXPORT_DIR, 'best_model_stage1.keras'),
    monitor='val_f1_macro',
    mode='max',
    save_best_only=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_f1_macro',
    mode='max',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

# Train stage 1
history_stage1 = stage1_model.fit(
    train_ds,
    epochs=FROZEN_EPOCHS,
    validation_data=test_ds,
    callbacks=[metrics_callback, early_stop, checkpoint, reduce_lr],
    verbose=1
)

# ─── STAGE 2: FINE TUNING ───────────────────────────────────────────────────
print("\n=== Stage 2: Fine-tuning Model ===")
# Load best stage 1 model
best_stage1_model = tf.keras.models.load_model(
    os.path.join(EXPORT_DIR, 'best_model_stage1.keras'),
    custom_objects={'FocalLoss': FocalLoss}  # Important: provide custom objects mapping
)

# Create fine-tuning model
stage2_model = build_model(trainable_base=True, fine_tuning=True)

# Copy weights from stage 1
stage2_model.set_weights(best_stage1_model.get_weights())

# Callbacks for stage 2
metrics_callback_stage2 = MetricsCallback(test_ds, ROAD_CLASSES, df_test)
early_stop_stage2 = EarlyStopping(
    monitor='val_f1_macro',
    mode='max',
    patience=16,
    restore_best_weights=True,
    verbose=1
)
checkpoint_stage2 = ModelCheckpoint(
    os.path.join(EXPORT_DIR, 'best_model_final.keras'),
    monitor='val_f1_macro',
    mode='max',
    save_best_only=True,
    verbose=1
)
reduce_lr_stage2 = ReduceLROnPlateau(
    monitor='val_f1_macro',
    mode='max',
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1
)

# Train stage 2
history_stage2 = stage2_model.fit(
    train_ds,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=test_ds,
    callbacks=[metrics_callback_stage2, early_stop_stage2, checkpoint_stage2, reduce_lr_stage2],
    verbose=1
)

# ─── EXPORT TO TFLITE ────────────────────────────────────────────────────────
print("\n=== Converting to TFLite for RP2040 Deployment ===")

# Load the best model
final_model = tf.keras.models.load_model(
    os.path.join(EXPORT_DIR, 'best_model_final.keras'),
    custom_objects={'FocalLoss': FocalLoss}  # Provide custom objects mapping here too
)

# Summary of the model
final_model.summary()

# Define a representative dataset for quantization
def representative_dataset():
    """Generate better representative dataset for quantization"""
    # Use stratified sampling to ensure all classes are represented
    class_specific_samples = {}

    # Get samples for each class
    for class_idx, class_name in enumerate(ROAD_CLASSES):
        # Find samples with this class
        class_samples = df_train[df_train['target_bin'].apply(
            lambda x: x.split(',')[class_idx] == '1'
        )]

        # If we have samples for this class, store them
        if len(class_samples) > 0:
            class_specific_samples[class_name] = class_samples

    # Generate calibration data
    for _ in range(50):  # First 50 samples - balanced across classes
        for class_name, samples in class_specific_samples.items():
            if len(samples) > 0:
                row = samples.sample(1).iloc[0]
                img_path = os.path.join(DATA_DIR, row['filename'])
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype(np.float32) / 255.0
                yield [np.expand_dims(img, axis=0)]

    # Add 50 random samples to improve representativeness
    for _ in range(50):
        row = df_train.sample(1).iloc[0]
        img_path = os.path.join(DATA_DIR, row['filename'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        yield [np.expand_dims(img, axis=0)]

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)

# Apply optimizations for RP2040
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert model
tflite_model = converter.convert()

# Save TFLite model
tflite_path = os.path.join(EXPORT_DIR, f"{NICKNAME}.tflite")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved: {tflite_path}")
print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")

# Also save a metadata file with class names for inference
class_names = {i: name for i, name in enumerate(ROAD_CLASSES)}
with open(os.path.join(EXPORT_DIR, "class_names.txt"), 'w') as f:
    for i, name in class_names.items():
        f.write(f"{i}: {name}\n")

print("Training complete! The optimized TFLite model is ready for deployment on the SparkFun Thing Plus RP2040.")