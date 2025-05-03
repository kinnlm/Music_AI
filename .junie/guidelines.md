# Music AI: WAV-to-MIDI Transcription System - Developer Guidelines

This document provides essential information for developers working on the Music AI project, a deep learning-based system that converts raw WAV audio into symbolic MIDI note events using a CNN + Transformer architecture.

## Build/Configuration Instructions

### Environment Setup

1. **Python Version**: This project requires Python 3.8+ with PyTorch 2.0+.

2. **Virtual Environment**: 
   ```bash
   # Create a new virtual environment
   python -m venv .venv_new
   
   # Activate the environment
   # On Windows:
   .venv_new\Scripts\activate
   # On Unix/MacOS:
   source .venv_new/bin/activate
   ```

3. **Dependencies**: Install the required packages:
   ```bash
   pip install torch torchvision torchaudio
   pip install librosa pretty_midi pandas numpy matplotlib optuna pyloudnorm
   pip install jupyter
   ```

4. **GPU Configuration**: 
   - The project benefits significantly from GPU acceleration.
   - Set environment variables for optimal GPU memory usage:
     ```python
     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
     os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
     ```

### Data Preparation

1. **MAESTRO Dataset**:
   - Download the MAESTRO dataset v3.0.0 from [the official site](https://magenta.tensorflow.org/datasets/maestro).
   - Extract it to the `maestro-v3.0.0` directory in the project root.

2. **Preprocessing**:
   - Run `Data_Preprocessing_Fixed.ipynb` to generate spectrograms and MIDI token sequences.
   - For testing with a smaller dataset, modify the notebook to process only a subset:
     ```python
     df = df.sample(n=5)  # Process only 5 files
     ```

3. **Directory Structure**:
   - Ensure these directories exist (they will be created by the preprocessing script if needed):
     ```
     spectrogram_cache/  # For storing processed spectrograms
     cnn_outputs/        # For storing CNN feature outputs
     checkpoints/        # For storing model checkpoints
     ```

## Testing Information

### Running Tests

1. **Model Testing**:
   - Use the `test_model.py` script to verify model functionality:
     ```bash
     python test_model.py
     ```
   - This script tests:
     - Model creation and loading
     - Inference with dummy input

2. **Custom Tests**:
   - Create test scripts in the project root directory
   - Import model definitions from `model_definitions.py`
   - Follow the pattern in `test_model.py` for consistent testing

### Adding New Tests

1. **Test File Structure**:
   ```python
   import torch
   from model_definitions import WAVtoMIDIModel
   
   def test_specific_functionality():
       # Test setup
       model = WAVtoMIDIModel(input_channels=11, d_model=512)
       
       # Test execution
       result = model(test_input)
       
       # Test verification
       assert result["pitch"].shape == expected_shape
       
       return True  # Return success status
   
   if __name__ == "__main__":
       success = test_specific_functionality()
       print(f"Test {'passed' if success else 'failed'}")
   ```

2. **Testing Data Processing**:
   - For testing data processing, use a small subset of the MAESTRO dataset
   - Example test for spectrogram generation:
   
   ```python
   import torch
   import librosa
   import numpy as np
   from pathlib import Path
   
   def test_spectrogram_generation():
       # Load a sample WAV file
       audio_path = Path("maestro-v3.0.0") / "2004" / "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"
       y, sr = librosa.load(audio_path, sr=None)
       
       # Generate spectrogram
       spectrogram = np.abs(librosa.cqt(y, sr=sr, hop_length=512, n_bins=192, bins_per_octave=24))
       
       # Verify shape and values
       assert spectrogram.shape[0] == 192, f"Expected 192 frequency bins, got {spectrogram.shape[0]}"
       assert np.isfinite(spectrogram).all(), "Spectrogram contains non-finite values"
       
       return True
   
   if __name__ == "__main__":
       success = test_spectrogram_generation()
       print(f"Spectrogram test {'passed' if success else 'failed'}")
   ```

### Example Test Execution

Here's a demonstration of running the model test:

```
$ python test_model.py
Running model tests...
Testing model loading...
Model created: WAVtoMIDIModel
Checkpoints directory exists: checkpoints
Found 0 checkpoint files
Testing model inference with dummy input...
Model inference successful!
Output keys: ['pitch', 'velocity', 'duration', 'time']
  pitch shape: torch.Size([1, 100, 128])
  velocity shape: torch.Size([1, 100, 32])
  duration shape: torch.Size([1, 100, 64])
  time shape: torch.Size([1, 100, 64])
Test Summary:
Model Loading: ✅ Passed
Model Inference: ✅ Passed
All tests passed! The model is working correctly.
```

## Additional Development Information

### Code Style

1. **Documentation**:
   - Use docstrings for all classes and functions
   - Follow Google-style docstring format
   - Include parameter types and return types

2. **Naming Conventions**:
   - Classes: CamelCase (e.g., `WAVtoMIDIModel`)
   - Functions/Methods: snake_case (e.g., `process_wav_to_spectrogram`)
   - Variables: snake_case (e.g., `spectrogram_cache_dir`)
   - Constants: UPPER_CASE (e.g., `MAX_T`)

3. **Code Organization**:
   - Keep notebook cells organized by functionality
   - Extract reusable functions to Python modules
   - Use comments to explain complex operations

### Memory Management

1. **GPU Memory**:
   - The project processes large spectrograms that can consume significant GPU memory
   - Use chunking for processing long sequences:
     ```python
     if spectrogram.shape[-1] > MAX_T:
         chunks = chunk_tensor(spectrogram, MAX_T, overlap=CHUNK_OVERLAP)
         # Process chunks individually
     ```
   
   - Clear GPU cache regularly:
     ```python
     torch.cuda.empty_cache()
     gc.collect()
     ```

2. **Batch Processing**:
   - Adjust batch sizes based on available memory
   - Monitor memory usage and reduce batch size if needed:
     ```python
     if get_available_ram() < memory_threshold:
         batch_size = max(min_batch_size, int(batch_size * 0.8))
     ```

### Model Architecture

1. **CNN Feature Extractor**:
   - `CNNFeatureSequence`: Processes spectrograms using residual blocks
   - Input: Spectrograms with shape [B, C=11, F, T]
   - Output: Feature sequences with shape [B, T, D]

2. **Transformer Encoder**:
   - Processes CNN features to capture temporal relationships
   - Uses multi-head self-attention mechanism

3. **Multi-Head Output**:
   - Predicts multiple attributes of MIDI notes:
     - Pitch (128 classes)
     - Velocity (32 bins)
     - Duration (64 bins)
     - Time since last note (64 bins)

### Hyperparameter Optimization

1. **Optuna Framework**:
   - The project uses Optuna for hyperparameter optimization
   - Study is stored in `optuna_study_music_model.db`
   - Key hyperparameters:
     - CNN depths
     - Transformer layers
     - Number of attention heads
     - Learning rate

2. **Custom Evaluation Metrics**:
   - `compute_soft_streak_score`: Evaluates model performance with emphasis on consecutive correct predictions
   - `compute_loss_classical`: Computes weighted cross-entropy loss for each output head

### Debugging Tips

1. **Common Issues**:
   - CUDA out of memory: Reduce batch size or use chunking
   - Invalid spectrograms: Check for NaN values or incorrect shapes
   - Slow processing: Use GPU acceleration and optimize preprocessing

2. **Logging**:
   - The code uses print statements for logging
   - Consider implementing a proper logging system for production

3. **Checkpointing**:
   - Save model checkpoints regularly during training
   - Save the best model based on validation metrics