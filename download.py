import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from PIL import Image

# --------------------------------------------------
# Configuration
# --------------------------------------------------

OUTPUT_DIR = Path("mnist")
SAMPLE_DIR = OUTPUT_DIR / "sample_10"

# Create folders
OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load MNIST from TensorFlow Datasets
# --------------------------------------------------

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=False,
    as_supervised=True,
    with_info=True,
)

print(f"Training images: {ds_info.splits['train'].num_examples}")
print(f"Test images:     {ds_info.splits['test'].num_examples}")

# --------------------------------------------------
# Export dataset
# --------------------------------------------------

def save_dataset(dataset, split_name):
    count = 0

    for image, label in tfds.as_numpy(dataset):
        label = int(label)

        # Create label folder
        label_dir = OUTPUT_DIR / split_name / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)

        # MNIST images are 28x28x1
        image = image.squeeze()

        # Convert to PNG
        img = Image.fromarray(image)

        img.save(label_dir / f"{count:05d}.png")

        count += 1

        if count % 1000 == 0:
            print(f"{split_name}: saved {count} images")

    print(f"{split_name}: DONE - {count} images")


# Save all training images
save_dataset(ds_train, "train")

# Save all test images
save_dataset(ds_test, "test")

# --------------------------------------------------
# Save 10 sample images from test set
# --------------------------------------------------

for i, (image, label) in enumerate(tfds.as_numpy(ds_test.take(10))):
    label = int(label)

    image = image.squeeze()
    img = Image.fromarray(image)

    # Include label in filename
    img.save(SAMPLE_DIR / f"{i:02d}_label_{label}.png")

print("\nDone!")
print(f"Dataset saved to: {OUTPUT_DIR.resolve()}")
print(f"10 samples saved to: {SAMPLE_DIR.resolve()}")