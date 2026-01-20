"""
Deploy custom trained YOLOv8 model for retail product detection
Run this after downloading your trained model from Colab
"""

import os
import shutil
from pathlib import Path

def deploy_custom_model():
    """
    Deploy custom trained retail detection model
    """
    print("🚀 Deploying Custom Retail Detection Model")
    print("=" * 50)
    
    # Paths
    project_root = Path("/Users/keerthiv.c/Desktop/msme")
    ai_service = project_root / "ai_service"
    models_dir = ai_service / "models"
    
    # Check if model exists in Downloads
    downloads = Path.home() / "Downloads"
    model_files = list(downloads.glob("*best.pt")) + list(downloads.glob("retail*.pt"))
    
    if not model_files:
        print("\n❌ No trained model found in Downloads folder")
        print("\nPlease:")
        print("1. Download 'best.pt' from Google Colab")
        print("2. Save it to your Downloads folder")
        print("3. Run this script again")
        return False
    
    # Use most recent model
    model_file = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"\n✓ Found model: {model_file.name}")
    
    # Create models directory
    models_dir.mkdir(exist_ok=True)
    print(f"✓ Models directory: {models_dir}")
    
    # Copy model
    dest_path = models_dir / "retail_yolov8s_best.pt"
    shutil.copy(model_file, dest_path)
    print(f"✓ Model copied to: {dest_path}")
    
    # Verify model
    try:
        from ultralytics import YOLO
        model = YOLO(str(dest_path))
        print(f"✓ Model loaded successfully")
        print(f"  - Classes: {len(model.names)}")
        print(f"  - First 5 classes: {list(model.names.values())[:5]}")
    except Exception as e:
        print(f"⚠ Warning: Could not verify model: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Deployment Complete!")
    print("\nNext steps:")
    print("1. Restart AI service")
    print("2. Test with shelf images")
    print("\nCommands:")
    print("  lsof -ti:8000 | xargs kill -9")
    print("  cd ai_service && python main.py")
    
    return True

if __name__ == "__main__":
    deploy_custom_model()
