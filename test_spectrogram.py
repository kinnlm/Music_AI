import torch
import librosa
import numpy as np
from pathlib import Path

def test_spectrogram_generation():
    """
    Test spectrogram generation from a WAV file.
    
    This test loads a sample WAV file from the MAESTRO dataset and generates
    a Constant-Q Transform (CQT) spectrogram. It then verifies that the 
    spectrogram has the expected shape and contains valid values.
    
    Returns:
        bool: True if the test passes, False otherwise.
    """
    try:
        # Find a WAV file in the MAESTRO dataset
        maestro_dir = Path("maestro-v3.0.0")
        if not maestro_dir.exists():
            print(f"❌ MAESTRO dataset directory not found: {maestro_dir}")
            return False
            
        # Look for WAV files in the dataset
        wav_files = list(maestro_dir.glob("**/*.wav"))
        if not wav_files:
            print(f"❌ No WAV files found in {maestro_dir}")
            return False
            
        # Use the first WAV file found
        audio_path = wav_files[0]
        print(f"Using audio file: {audio_path}")
        
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        print(f"Audio loaded: {len(y)} samples, {sr} Hz")
        
        # Generate spectrogram using CQT
        print("Generating CQT spectrogram...")
        spectrogram = np.abs(librosa.cqt(y, sr=sr, hop_length=512, n_bins=192, bins_per_octave=24))
        
        # Verify shape and values
        print(f"Spectrogram shape: {spectrogram.shape}")
        assert spectrogram.shape[0] == 192, f"Expected 192 frequency bins, got {spectrogram.shape[0]}"
        assert np.isfinite(spectrogram).all(), "Spectrogram contains non-finite values"
        
        print("✅ Spectrogram generation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in spectrogram generation test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running spectrogram generation test...")
    success = test_spectrogram_generation()
    print(f"Spectrogram test {'passed' if success else 'failed'}")