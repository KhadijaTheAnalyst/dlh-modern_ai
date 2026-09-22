#!/usr/bin/env python3
"""
Hyperparameter tuning and final training for YOLOv8 object detection.

Performs two-phase training pipeline:
Phase 1: Lightweight hyperparameter search to find optimal settings
Phase 2: Continue training from best checkpoint to convergence
"""

from ultralytics import YOLO
import os


def tune_hyperparameters():
    """
    Performs two-phase hyperparameter tuning and training.

    Phase 1: Lightweight hyperparameter search using YOLO's tune() method
             - Tests 15-20 configurations
             - 10 epochs per trial for efficiency
             - Explores: learning rates, augmentation, geometric transforms,
               loss weights, optimizer settings
             - Saves best hyperparameters and model to: runs/detect/tune/

    Phase 2: Continue training from best checkpoint to convergence
             - Loads best model from Phase 1
             - Trains for ~150 total epochs
             - Applies optimal hyperparameters discovered
             - Uses early stopping to prevent overfitting
             - Saves final model as: best_model.pt

    Returns:
        None (saves final model to best_model.pt)

    Expected Performance:
        - mAP50 ≥ 65%
        - mAP50-95 ≥ 46%
    """

    # ==================== PHASE 1: HYPERPARAMETER TUNING ====================
    print("=" * 70)
    print("PHASE 1: LIGHTWEIGHT HYPERPARAMETER TUNING")
    print("=" * 70)
    print("Testing 15-20 hyperparameter configurations...")
    print("This phase explores optimal settings for learning rates,")
    print("augmentation, and other training parameters using short")
    print("10-epoch trials.\n")

    # Load base YOLO model
    model = YOLO("yolov8n.pt")

    # Perform hyperparameter tuning
    # YOLO automatically tests different hyperparameter combinations
    # and saves the best configuration
    results = model.tune(
        data="datasets/detection/data.yaml",
        epochs=10,  # Short trials for efficiency
        iterations=20,  # Test 20 different hyperparameter configurations
        imgsz=640,
        batch=8,
        device=0,
        verbose=False,
        plots=True,
        patience=5,
        seed=42
    )

    print("✅ Phase 1 Complete: Optimal hyperparameters discovered")
    print(f"   Results saved to: runs/detect/tune/")
    print(f"   Best model: runs/detect/tune/weights/best.pt")
    print(f"   Best hyperparams: runs/detect/tune/best_hyperparameters.yaml\n")

    # ==================== PHASE 2: FINAL TRAINING ====================
    print("=" * 70)
    print("PHASE 2: FINAL TRAINING TO CONVERGENCE")
    print("=" * 70)
    print(
        "Continuing training from best checkpoint with optimal "
        "hyperparameters..."
    )
    print("Training for ~150 total epochs with early stopping.\n")

    # Load the best model from tuning phase
    best_tune_model_path = "runs/detect/tune/weights/best.pt"

    if os.path.exists(best_tune_model_path):
        model = YOLO(best_tune_model_path)
        print(f"✅ Loaded best model from: {best_tune_model_path}")
    else:
        print("⚠️  Warning: Best tune model not found")
        print("   Using base yolov8n.pt for training")
        model = YOLO("yolov8n.pt")

    # Train model to convergence with early stopping
    # The optimal hyperparameters from Phase 1 are automatically used
    # if the model was loaded from the tuned checkpoint
    final_results = model.train(
        data="datasets/detection/data.yaml",
        epochs=150,  # Target total of ~150 epochs
        imgsz=640,
        batch=8,
        device=0,
        patience=20,  # Early stopping: stop if no improvement after 20 epochs
        plots=True,  # Generate training plots
        save=True,   # Save checkpoints
        verbose=False,
        seed=42
    )

    # Save final trained model
    final_model_path = "best_model.pt"
    model.save(final_model_path)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {final_model_path}\n")

    # Display final performance metrics
    if final_results and hasattr(final_results, 'results_dict'):
        metrics = final_results.results_dict

        # Extract key metrics
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)

        print("Performance Metrics:")
        print(
            f"  mAP50:     {map50:.4f} ({map50*100:.1f}%) - "
            "Target: ≥ 0.65 (65%)"
        )
        print(
            f"  mAP50-95:  {map50_95:.4f} ({map50_95*100:.1f}%) - "
            "Target: ≥ 0.46 (46%)"
        )
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}\n")

        # Check if targets are met
        target_map50 = 0.65
        target_map50_95 = 0.46

        if map50 >= target_map50 and map50_95 >= target_map50_95:
            print("✨ SUCCESS! Performance targets achieved!")
        else:
            print("⚠️  Performance targets not yet met:")
            if map50 < target_map50:
                gap = (target_map50 - map50) * 100
                print(f"   - mAP50: Need {gap:.1f}% improvement")
            if map50_95 < target_map50_95:
                gap = (target_map50_95 - map50_95) * 100
                print(f"   - mAP50-95: Need {gap:.1f}% improvement")
            print("\n   Consider:")
            print("   - Training on GPU for faster convergence")
            print("   - Increasing epochs beyond 150")
            print("   - Fine-tuning discovered hyperparameters")
            print("   - Using larger model (yolov8s.pt, yolov8m.pt)")

    print("\nTraining plots saved to: runs/detect/train/")


if __name__ == "__main__":
    tune_hyperparameters()
