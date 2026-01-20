"""
Test the downloaded pre-trained retail model
"""

from ultralytics import YOLO

# Load the pre-trained retail model
model = YOLO('models/retail_yolov8_best.pt')

print("✓ Pre-trained retail model loaded successfully!")
print(f"\nModel Information:")
print(f"  - Number of classes: {len(model.names)}")
print(f"  - Model type: {model.model.__class__.__name__}")
print(f"\nFirst 20 product classes:")
for i, (key, name) in enumerate(list(model.names.items())[:20]):
    print(f"  {key}: {name}")

print(f"\n... and {len(model.names) - 20} more classes")
print("\n✅ Model is ready to use for retail product detection!")
