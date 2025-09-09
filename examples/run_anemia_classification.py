#!/usr/bin/env python3
"""Example script for anemia classification."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bloodwork_ai.tasks.cbc.anemia_classifier import AnemiaClassifier
from bloodwork_ai.utils.log import setup_logging


def main():
    """Run anemia classification example."""
    # Setup logging
    setup_logging(log_level="INFO")
    
    print("🚀 Bloodwork AI - Anemia Classification Example")
    print("=" * 50)
    
    # Initialize classifier
    classifier = AnemiaClassifier()
    
    # Check if sample data exists
    data_path = Path("sample_data/tabular/cbc_data.csv")
    if not data_path.exists():
        print("❌ Sample data not found. Please run:")
        print("   python scripts/generate_sample_data.py")
        return
    
    print("📊 Loading and preparing data...")
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(
        str(data_path),
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )
    
    print(f"✅ Data prepared:")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Validation samples: {len(X_val)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Features: {len(X_train.columns)}")
    
    # Train model
    print("\n🤖 Training model...")
    training_results = classifier.train(
        X_train, y_train, X_val, y_val, algorithm="xgboost"
    )
    
    print(f"✅ Model trained successfully!")
    print(f"   - Algorithm: {training_results['algorithm']}")
    print(f"   - Features: {training_results['feature_count']}")
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    evaluation_results = classifier.evaluate(X_test, y_test)
    
    metrics = evaluation_results["metrics"]
    print(f"✅ Evaluation completed:")
    print(f"   - Accuracy: {metrics.get('accuracy', 0):.3f}")
    print(f"   - F1 Score: {metrics.get('f1_macro', 0):.3f}")
    print(f"   - ROC AUC: {metrics.get('roc_auc_ovr', 0):.3f}")
    
    # Show feature importance
    print("\n🔍 Top 10 Most Important Features:")
    feature_importance = classifier.get_feature_importance()
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f"   {i+1:2d}. {feature}: {importance:.4f}")
    
    # Make sample predictions
    print("\n🔮 Sample Predictions:")
    sample_predictions = classifier.predict(X_test.head(5))
    sample_probs = classifier.predict_proba(X_test.head(5))
    
    for i, (pred, prob) in enumerate(zip(sample_predictions, sample_probs)):
        true_label = y_test.iloc[i]
        pred_label = pred
        confidence = max(prob)
        print(f"   Sample {i+1}: True={true_label}, Predicted={pred_label}, Confidence={confidence:.3f}")
    
    # Save model
    model_path = "artifacts/models/anemia_classifier.pkl"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    print("\n🎉 Anemia classification example completed successfully!")
    print("\nNext steps:")
    print("1. Run the WBC detection example: python examples/run_wbc_detection.py")
    print("2. Start the API server: make serve")
    print("3. View MLflow UI: make mlflow-ui")


if __name__ == "__main__":
    main()
