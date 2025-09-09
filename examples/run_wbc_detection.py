#!/usr/bin/env python3
"""Example script for WBC detection."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bloodwork_ai.models.vision import YOLODetector
from bloodwork_ai.utils.log import setup_logging


def main():
    """Run WBC detection example."""
    # Setup logging
    setup_logging(log_level="INFO")
    
    print("🚀 Bloodwork AI - WBC Detection Example")
    print("=" * 50)
    
    # Check if sample data exists
    images_dir = Path("sample_data/images/images")
    if not images_dir.exists():
        print("❌ Sample images not found. Please run:")
        print("   python scripts/generate_sample_data.py")
        return
    
    # Initialize detector
    detector = YOLODetector()
    
    # For this example, we'll use a pre-trained YOLO model
    # In practice, you would train your own model
    print("📦 Loading pre-trained YOLO model...")
    try:
        # Try to load a pre-trained model
        detector.load_model("yolov8n.pt")
        print("✅ Pre-trained model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load pre-trained model: {e}")
        print("   This is expected if you don't have a trained model yet.")
        print("   In a real scenario, you would train the model first.")
        return
    
    # Get sample images
    image_files = list(images_dir.glob("*.jpg"))
    if not image_files:
        print("❌ No sample images found in the directory")
        return
    
    print(f"📸 Found {len(image_files)} sample images")
    
    # Process first few images
    sample_images = image_files[:3]
    
    print("\n🔍 Processing sample images...")
    
    for i, image_path in enumerate(sample_images):
        print(f"\n   Image {i+1}: {image_path.name}")
        
        try:
            # Load and process image
            image = detector.load_image(str(image_path))
            
            # Make predictions
            detections = detector.predict(image)
            
            print(f"      Detections: {len(detections)}")
            
            if detections:
                # Count cells by type
                cell_counts = detector.count_cells(image)
                print(f"      Cell counts: {cell_counts}")
                
                # Get statistics
                stats = detector.get_cell_statistics(image)
                print(f"      Total cells: {stats['total_cells']}")
                print(f"      Avg confidence: {stats['avg_confidence']:.3f}")
                print(f"      Avg area: {stats['avg_area']:.1f} pixels")
            
        except Exception as e:
            print(f"      ❌ Error processing image: {e}")
    
    # Demonstrate visualization
    print("\n🎨 Creating visualization...")
    try:
        # Process first image for visualization
        image_path = sample_images[0]
        image = detector.load_image(str(image_path))
        detections = detector.predict(image)
        
        # Create visualization
        vis_image = detector.visualize_predictions(
            image, 
            detections,
            show_labels=True,
            show_conf=True
        )
        
        # Save visualization
        output_path = "artifacts/wbc_detection_visualization.jpg"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        detector.visualize_predictions(
            image, 
            detections,
            save_path=output_path,
            show_labels=True,
            show_conf=True
        )
        
        print(f"✅ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
    
    print("\n🎉 WBC detection example completed successfully!")
    print("\nNext steps:")
    print("1. Train your own model with real data")
    print("2. Start the API server: make serve")
    print("3. View MLflow UI: make mlflow-ui")


if __name__ == "__main__":
    main()
