"""
Road Object Detection Testing Script
For TFLite model evaluation on RP2040 deployment
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random
from collections import defaultdict
import pandas as pd

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
EXPORT_DIR = "../TensorFlow/export"
TFLITE_MODEL_PATH = os.path.join(EXPORT_DIR, "RoadLiteMobileNetV2.tflite")
DATA_DIR = "../TensorFlow/Data"
PREDICTION_DIR = "./predictions"
IMG_HEIGHT, IMG_WIDTH = 64, 64
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
EXCEL_TEST_PATH = "../TensorFlow/excel/test.xlsx"  # Path to test dataset info

# Load the class names from the export directory
CLASS_NAMES = []
with open(os.path.join(EXPORT_DIR, "class_names.txt"), 'r') as f:
    for line in f:
        parts = line.strip().split(': ', 1)
        if len(parts) == 2:
            CLASS_NAMES.append(parts[1])
NUM_CLASSES = len(CLASS_NAMES)

# Create output directory
os.makedirs(PREDICTION_DIR, exist_ok = True)

# ─── LOAD TFLITE INTERPRETER ─────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path = TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

print(f"TFLite model loaded and ready for inference")
print(f"Input shape: {input_shape}")
print(f"Input dtype: {input_dtype}")
print(f"Detecting {NUM_CLASSES} classes: {', '.join(CLASS_NAMES)}")

# ─── LOAD THRESHOLDS ─────────────────────────────────────────────────────────
thresholds_path = os.path.join(EXPORT_DIR, "best_thresholds.npy")
if os.path.exists(thresholds_path):
    thresholds = np.load(thresholds_path)
    print(f"Loaded optimized thresholds from {thresholds_path}: {thresholds}")
else:
    thresholds = np.array([0.5] * NUM_CLASSES)
    print("Using default thresholds of 0.5")


# ─── PREPROCESS FUNCTION ─────────────────────────────────────────────────────
def preprocess_image(img_path):
    """Preprocess image for model inference with proper handling for quantized models"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Handle different input types based on model quantization
    if input_dtype == np.int8:
        # For quantized models, need to properly quantize input
        scale, zero_point = input_details[0]['quantization']
        img = img.astype(np.float32)
        img = img / 255.0  # Normalize to [0,1]
        img = img / scale + zero_point  # Apply quantization params
        img = np.clip(img, -128, 127).astype(np.int8)  # Clip and convert to int8
    elif input_dtype == np.uint8:
        # For uint8 quantized models
        img = img.astype(np.uint8)
    else:
        # For float models
        img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis = 0)  # Add batch dimension
    return img


# ─── INFERENCE FUNCTION ──────────────────────────────────────────────────────
def run_inference(img_path):
    """Run inference on an image and return raw outputs and thresholded predictions"""
    img = preprocess_image(img_path)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # If output is quantized, dequantize it
    if output_details[0]['dtype'] == np.int8 or output_details[0]['dtype'] == np.uint8:
        # Get quantization parameters
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    # Ensure outputs are in range [0,1] for confidence scores (apply sigmoid if needed)
    if np.any(output < 0) or np.any(output > 1):
        output = 1 / (1 + np.exp(-output))  # Apply sigmoid if outputs aren't already sigmoids

    # Apply thresholds for binary predictions
    preds = (output > thresholds).astype(int)

    return output, preds


# ─── EVALUATE ON TEST DATASET ─────────────────────────────────────────────────
def evaluate_test_dataset():
    """Evaluate model on test dataset with labeled ground truth"""
    # Load test dataset info if available
    if not os.path.exists(EXCEL_TEST_PATH):
        print(f"⚠️ Test dataset info not found at {EXCEL_TEST_PATH}")
        return None, None, None, None

    try:
        df_test = pd.read_excel(EXCEL_TEST_PATH)
        print(f"Loaded test dataset with {len(df_test)} samples")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return None, None, None, None

    # Prepare for evaluation
    y_true = []
    y_pred = []
    y_scores = []
    filenames = []

    # Use a subset for faster evaluation if dataset is large
    max_eval_samples = 500
    if len(df_test) > max_eval_samples:
        df_test = df_test.sample(max_eval_samples, random_state = 42)
        print(f"Using {max_eval_samples} random samples for evaluation")

    # Process each test sample
    for idx, row in df_test.iterrows():
        try:
            filename = row['filename']
            filenames.append(filename)
            img_path = os.path.join(DATA_DIR, filename)

            # Get ground truth
            true_labels = list(map(int, row['target_bin'].split(',')))
            y_true.append(true_labels)

            # Run inference
            scores, preds = run_inference(img_path)
            y_scores.append(scores)
            y_pred.append(preds)

            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(df_test)} test samples")

        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    return y_true, y_pred, y_scores, filenames


# ─── CALCULATE METRICS ─────────────────────────────────────────────────────────
def calculate_metrics(y_true, y_pred):
    """Calculate and print evaluation metrics"""
    if y_true is None or y_pred is None:
        return

    # Print full classification report
    report = classification_report(
        y_true, y_pred,
        target_names = CLASS_NAMES,
        zero_division = 0,
        output_dict = False
    )
    print("\n===== Classification Report =====")
    print(report)

    # Also get report as dictionary for visualization
    report_dict = classification_report(
        y_true, y_pred,
        target_names = CLASS_NAMES,
        zero_division = 0,
        output_dict = True
    )

    # Save the report to a file
    with open(os.path.join(PREDICTION_DIR, "evaluation_report.txt"), 'w') as f:
        f.write("===== Classification Report =====\n")
        f.write(report)

    return report_dict


# ─── VISUALIZE METRICS ─────────────────────────────────────────────────────────
def visualize_metrics(report_dict):
    """Create visualizations of model performance metrics"""
    if report_dict is None:
        return

    # Extract class metrics
    class_metrics = {}
    for cls in CLASS_NAMES:
        if cls in report_dict:
            class_metrics[cls] = {
                'precision':report_dict[cls]['precision'],
                'recall':report_dict[cls]['recall'],
                'f1-score':report_dict[cls]['f1-score'],
                'support':report_dict[cls]['support']
            }

    # Create metrics dataframe
    metrics_df = pd.DataFrame(class_metrics).T
    metrics_df = metrics_df.reset_index().rename(columns = {'index':'class'})

    # 1. Bar chart of F1 scores
    plt.figure(figsize = (10, 6))
    ax = sns.barplot(x = 'class', y = 'f1-score', data = metrics_df, palette = 'viridis')
    ax.set_title('F1 Scores by Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('F1 Score')
    plt.xticks(rotation = 45, ha = 'right')
    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_DIR, "f1_scores.png"))

    # 2. Precision vs Recall plot
    plt.figure(figsize = (10, 6))
    plt.scatter(metrics_df['precision'], metrics_df['recall'], s = metrics_df['support'], alpha = 0.7)
    for i, cls in enumerate(metrics_df['class']):
        plt.annotate(cls,
                     (metrics_df['precision'][i], metrics_df['recall'][i]),
                     xytext = (5, 5),
                     textcoords = 'offset points')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall by Class (circle size = sample count)')
    plt.grid(True, linestyle = '--', alpha = 0.7)
    # Add reference lines
    plt.axhline(y = 0.5, color = 'r', linestyle = ':', alpha = 0.5)
    plt.axvline(x = 0.5, color = 'r', linestyle = ':', alpha = 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_DIR, "precision_recall.png"))

    # 3. Confusion matrices (one-vs-rest for each class)
    fig, axes = plt.subplots(2, 4, figsize = (20, 10))
    axes = axes.flatten()

    for i, cls in enumerate(CLASS_NAMES):
        if i < len(axes):
            ax = axes[i]
            tp = report_dict[cls]['support'] * report_dict[cls]['recall']
            fn = report_dict[cls]['support'] - tp
            fp = (tp / report_dict[cls]['precision']) - tp if report_dict[cls]['precision'] > 0 else 0
            tn = report_dict['samples avg']['support'] - tp - fn - fp

            cm = np.array([[tn, fp], [fn, tp]])
            sns.heatmap(cm, annot = True, fmt = '.0f', cmap = 'Blues', ax = ax)
            ax.set_title(f'Confusion Matrix: {cls}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_DIR, "confusion_matrices.png"))


# ─── ANALYZE ERRORS ───────────────────────────────────────────────────────────
def analyze_errors(y_true, y_pred, y_scores, filenames):
    """Analyze and visualize prediction errors and edge cases"""
    if y_true is None or y_pred is None:
        return

    # Calculate error types for each sample
    errors = defaultdict(list)

    for i, (true, pred, scores, fname) in enumerate(zip(y_true, y_pred, y_scores, filenames)):
        # Check for false positives and false negatives
        for cls_idx, (t, p) in enumerate(zip(true, pred)):
            if t == 1 and p == 0:  # False negative
                errors['false_negatives'].append((fname, cls_idx, scores[cls_idx]))
            elif t == 0 and p == 1:  # False positive
                errors['false_positives'].append((fname, cls_idx, scores[cls_idx]))

        # Check for samples with low confidence (high entropy in predictions)
        if np.max(scores) < 0.6:
            errors['low_confidence'].append((fname, np.argmax(scores), np.max(scores)))

        # Check for samples with multiple high confidence predictions
        if sum(scores > 0.7) > 1:
            errors['multiple_detections'].append((fname, np.where(scores > 0.7)[0].tolist(),
                                                  scores[scores > 0.7].tolist()))

    # Print error statistics
    print("\n===== Error Analysis =====")
    print(f"False Negatives: {len(errors['false_negatives'])}")
    print(f"False Positives: {len(errors['false_positives'])}")
    print(f"Low Confidence: {len(errors['low_confidence'])}")
    print(f"Multiple Detections: {len(errors['multiple_detections'])}")

    # Visualize examples of each error type
    error_types = {
        'false_negatives':'False Negatives (missed detections)',
        'false_positives':'False Positives (incorrect detections)',
        'low_confidence':'Low Confidence Predictions',
        'multiple_detections':'Multiple High-Confidence Detections'
    }

    for error_type, title in error_types.items():
        if len(errors[error_type]) == 0:
            continue

        # Sample up to 8 examples of this error type
        samples = random.sample(errors[error_type], min(8, len(errors[error_type])))

        # Create figure
        fig, axes = plt.subplots(2, 4, figsize = (16, 8))
        axes = axes.flatten()

        for i, sample in enumerate(samples):
            if i < len(axes):
                fname = sample[0]
                img_path = os.path.join(DATA_DIR, fname)

                # Load and display image
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, IMG_SIZE)

                    axes[i].imshow(img)
                    axes[i].axis('off')

                    # Add appropriate caption based on error type
                    if error_type == 'false_negatives':
                        cls_idx = sample[1]
                        conf = sample[2]
                        axes[i].set_title(
                            f"Missed {CLASS_NAMES[cls_idx]}\nConf: {conf:.2f} < {thresholds[cls_idx]:.2f}")
                    elif error_type == 'false_positives':
                        cls_idx = sample[1]
                        conf = sample[2]
                        axes[i].set_title(f"False {CLASS_NAMES[cls_idx]}\nConf: {conf:.2f} > {thresholds[cls_idx]:.2f}")
                    elif error_type == 'low_confidence':
                        cls_idx = sample[1]
                        conf = sample[2]
                        axes[i].set_title(f"Low conf: {CLASS_NAMES[cls_idx]}\nConf: {conf:.2f}")
                    elif error_type == 'multiple_detections':
                        class_indices = sample[1]
                        confs = sample[2]
                        class_names = [CLASS_NAMES[idx] for idx in class_indices]
                        axes[i].set_title("\n".join([f"{cls}: {conf:.2f}" for cls, conf in zip(class_names, confs)]))

                except Exception as e:
                    print(f"Error displaying {fname}: {e}")
                    axes[i].text(0.5, 0.5, f"Error: {e}", ha = 'center', va = 'center')
                    axes[i].axis('off')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(PREDICTION_DIR, f"error_{error_type}.png"))


# ─── TEST RANDOM IMAGES ──────────────────────────────────────────────────────
def test_random_images(num_samples = 8):
    """Test and visualize model on random images"""
    image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print(f"No image files found in {DATA_DIR}")
        return

    # Sample randomly
    sampled_files = np.random.choice(image_files, size = min(num_samples, len(image_files)), replace = False)

    # Create figure with appropriate number of rows and columns
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize = (cols * 4, rows * 4))

    # Handle single axis case
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, fname in enumerate(sampled_files):
        img_path = os.path.join(DATA_DIR, fname)

        try:
            # Load original image for display
            raw_img = cv2.imread(img_path)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            raw_img = cv2.resize(raw_img, IMG_SIZE)

            # Run inference
            output, preds = run_inference(img_path)

            # Get predicted class names
            pred_labels = [CLASS_NAMES[j] for j in range(NUM_CLASSES) if preds[j]]
            if not pred_labels:
                pred_labels = ["None"]

            # Get top 3 predictions with scores
            top_indices = np.argsort(output)[::-1][:3]
            top_scores = [(CLASS_NAMES[j], output[j]) for j in top_indices]

            # Format label string
            label_str = "\n".join([f"{name}: {score:.2f} ({'✓' if score > thresholds[j] else '✗'})"
                                   for j, (name, score) in enumerate(top_scores)])

            # Display image and predictions
            axes[i].imshow(raw_img)
            axes[i].axis("off")
            axes[i].set_title(
                f"Predictions: {', '.join(pred_labels)}\n\n{label_str}",
                fontsize = 9
            )

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            axes[i].text(0.5, 0.5, f"Error: {e}", ha = 'center', va = 'center')
            axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_DIR, "test_predictions.png"))
    plt.close()


# ─── INFERENCE SPEED TEST ────────────────────────────────────────────────────
def test_inference_speed(num_runs = 50):
    """Test inference speed over multiple runs"""
    image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files or len(image_files) < 5:
        print(f"Not enough image files found in {DATA_DIR} for speed test")
        return

    # Sample some files for testing
    sampled_files = np.random.choice(image_files, size = min(5, len(image_files)), replace = False)

    # Preload images to avoid I/O overhead in timing
    preloaded_images = {}
    for fname in sampled_files:
        img_path = os.path.join(DATA_DIR, fname)
        img = preprocess_image(img_path)
        preloaded_images[fname] = img

    # Run inference multiple times and measure
    inference_times = []

    for _ in range(num_runs):
        # Select random image
        fname = np.random.choice(list(preloaded_images.keys()))
        img = preloaded_images[fname]

        # Time inference
        start_time = cv2.getTickCount()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img)
        # Run inference
        interpreter.invoke()
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        end_time = cv2.getTickCount()
        inference_time = (end_time - start_time) / cv2.getTickFrequency() * 1000  # ms
        inference_times.append(inference_time)

    # Calculate statistics
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)

    print("\n===== Inference Speed Test =====")
    print(f"Average inference time: {avg_time:.2f} ms ({1000 / avg_time:.2f} FPS)")
    print(f"Standard deviation: {std_time:.2f} ms")
    print(f"Min/Max time: {min_time:.2f}/{max_time:.2f} ms")

    # Plot histogram of inference times
    plt.figure(figsize = (10, 6))
    plt.hist(inference_times, bins = 20, alpha = 0.7, color = 'blue')
    plt.axvline(avg_time, color = 'red', linestyle = 'dashed', linewidth = 2, label = f'Mean: {avg_time:.2f} ms')
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Inference Time Distribution')
    plt.legend()
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.savefig(os.path.join(PREDICTION_DIR, "inference_speed.png"))
    plt.close()

    # Save results to file
    with open(os.path.join(PREDICTION_DIR, "inference_speed.txt"), 'w') as f:
        f.write("===== Inference Speed Test =====\n")
        f.write(f"Average inference time: {avg_time:.2f} ms ({1000 / avg_time:.2f} FPS)\n")
        f.write(f"Standard deviation: {std_time:.2f} ms\n")
        f.write(f"Min/Max time: {min_time:.2f}/{max_time:.2f} ms\n")


# ─── ANALYZE MODEL SIZE ────────────────────────────────────────────────────────
def analyze_model_size():
    """Analyze and report model size information"""
    model_size = os.path.getsize(TFLITE_MODEL_PATH)
    model_size_kb = model_size / 1024
    model_size_mb = model_size_kb / 1024

    print("\n===== Model Size Analysis =====")
    print(f"Model file: {TFLITE_MODEL_PATH}")
    print(f"Model size: {model_size_kb:.2f} KB ({model_size_mb:.2f} MB)")

    # RP2040 has limited flash (typically 2MB) and RAM (264KB), so provide context
    print(f"Percentage of typical RP2040 flash (2MB): {model_size_mb / 2:.2f}%")
    print(f"Percentage of typical RP2040 RAM (264KB): {model_size_kb / 264:.2f}%")

    # Save results to file
    with open(os.path.join(PREDICTION_DIR, "model_size.txt"), 'w') as f:
        f.write("===== Model Size Analysis =====\n")
        f.write(f"Model file: {TFLITE_MODEL_PATH}\n")
        f.write(f"Model size: {model_size_kb:.2f} KB ({model_size_mb:.2f} MB)\n")
        f.write(f"Percentage of typical RP2040 flash (2MB): {model_size_mb / 2:.2f}%\n")
        f.write(f"Percentage of typical RP2040 RAM (264KB): {model_size_kb / 264:.2f}%\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
      # Test on random images
    print("\n1. Testing on random images...")
    test_random_images(num_samples = 8)

    # Evaluate on test dataset
    print("\n2. Evaluating on test dataset...")
    y_true, y_pred, y_scores, filenames = evaluate_test_dataset()

    # Calculate and visualize metrics
    if y_true is not None:
        print("\n3. Calculating metrics...")
        report_dict = calculate_metrics(y_true, y_pred)

        print("\n4. Visualizing performance metrics...")
        visualize_metrics(report_dict)

        print("\n5. Analyzing errors...")
        analyze_errors(y_true, y_pred, y_scores, filenames)

    # Test inference speed
    print("\n6. Testing inference speed...")
    test_inference_speed()

    # Analyze model size
    print("\n7. Analyzing model size...")
    analyze_model_size()

    print(f"\nEvaluation complete. All results saved to {PREDICTION_DIR}/")


if __name__ == "__main__":
    main()