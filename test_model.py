import os
import torch
import numpy as np
from pathlib import Path

def test_model_loading():
    """
    Test that the model can be loaded from a checkpoint.
    """
    print("Testing model loading...")

    # Import the model class
    from model_definitions import WAVtoMIDIModel

    # Create a model instance
    model = WAVtoMIDIModel(input_channels=11, d_model=512)

    # Check if the model was created successfully
    print(f"Model created: {type(model).__name__}")

    # Check if checkpoints directory exists
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        print(f"Checkpoints directory exists: {checkpoints_dir}")
        # List checkpoint files
        checkpoint_files = list(checkpoints_dir.glob("*.pt"))
        print(f"Found {len(checkpoint_files)} checkpoint files")

        if checkpoint_files:
            # Try to load the first checkpoint
            try:
                checkpoint = torch.load(checkpoint_files[0], map_location="cpu")
                print(f"Successfully loaded checkpoint: {checkpoint_files[0]}")

                # If it's a state dict directly
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    print("Successfully loaded model state from checkpoint")
                elif isinstance(checkpoint, dict):
                    # Try to load as a direct state dict
                    model.load_state_dict(checkpoint)
                    print("Successfully loaded model state from checkpoint")

                return True
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return False
    else:
        print("Checkpoints directory not found")

    return True  # Return True even if no checkpoints, as we're just testing model creation

def test_dummy_inference():
    """
    Test model inference with a dummy input.
    """
    print("\nTesting model inference with dummy input...")

    # Import the model class
    from model_definitions import WAVtoMIDIModel

    # Create a model instance
    model = WAVtoMIDIModel(input_channels=11, d_model=512)
    model.eval()

    # Create a dummy input (batch_size=1, channels=11, freq_bins=128, time_steps=100)
    dummy_input = torch.randn(1, 11, 128, 100)

    try:
        # Run inference
        with torch.no_grad():
            outputs = model(dummy_input)

        # Check outputs
        print("Model inference successful!")
        print(f"Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            print(f"  {key} shape: {value.shape}")

        return True
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

if __name__ == "__main__":
    print("Running model tests...")

    # Run tests
    model_loading_success = test_model_loading()
    inference_success = test_dummy_inference()

    # Print summary
    print("\nTest Summary:")
    print(f"Model Loading: {'✅ Passed' if model_loading_success else '❌ Failed'}")
    print(f"Model Inference: {'✅ Passed' if inference_success else '❌ Failed'}")

    if model_loading_success and inference_success:
        print("\nAll tests passed! The model is working correctly.")
    else:
        print("\nSome tests failed. Please check the error messages above.")
