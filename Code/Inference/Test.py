import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
EXPORT_DIR = "../TensorFlow/export"
TFLITE_MODEL_PATH = os.path.join(EXPORT_DIR, "RoadLiteMobileNetV2.tflite")
DATA_DIR = "../TensorFlow/Data"
PREDICTION_DIR = "./predictions"
IMG_HEIGHT, IMG_WIDTH = 64, 64
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

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

print(f"✅ TFLite model loaded and ready for inference")
print(f"Input shape: {input_shape}")
print(f"Input dtype: {input_dtype}")
print(f"Detecting {NUM_CLASSES} classes: {', '.join(CLASS_NAMES)}")

# ─── LOAD THRESHOLDS ─────────────────────────────────────────────────────────
thresholds_path = os.path.join(EXPORT_DIR, "best_thresholds.npy")
if os.path.exists(thresholds_path):
    thresholds = np.load(thresholds_path)
    print(f"✅ Loaded optimized thresholds from {thresholds_path}")
else:
    thresholds = np.array([0.5] * NUM_CLASSES)
    print("⚠️ Using default thresholds of 0.5")


# ─── PREPROCESS FUNCTION ─────────────────────────────────────────────────────
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Handle different input types based on model quantization
    if input_dtype == np.uint8:
        # For quantized models
        img = img.astype(np.uint8)
    else:
        # For float models
        img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis = 0)  # Add batch dimension
    return img


# ─── INFERENCE FUNCTION ──────────────────────────────────────────────────────
def run_inference(img_path):
    img = preprocess_image(img_path)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # If output is quantized, dequantize it
    if output_details[0]['dtype'] == np.uint8:
        # Get quantization parameters
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    # Apply thresholds for binary predictions
    preds = (output > thresholds).astype(int)

    return output, preds


# ─── TEST RANDOM IMAGES ──────────────────────────────────────────────────────
def test_random_images(num_samples = 8):
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
            label_str = "\n".join([f"{name}: {score:.2f}" for name, score in top_scores])

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
    plt.show()


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Testing model: {TFLITE_MODEL_PATH}")
    test_random_images(num_samples = 8)
    print(f"✅ Test complete. Predictions saved to {os.path.join(PREDICTION_DIR, 'test_predictions.png')}")