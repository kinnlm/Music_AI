
# 🎼 WAV-to-MIDI Transformer Project Summary

## 🏗️ Model Architecture

### ✅ CNN Feature Extractor (`CNNFeatureSequence`)
- Input: Multi-channel spectrogram `[B, C, F, T]` (e.g. 11 channels)
- Output: Sequence of embeddings `[B, T, D]` for Transformer
- Based on ResNet-style blocks
- Designed to preserve **temporal structure**

### ✅ Transformer Encoder
- Input: CNN output sequence `[B, T, D]`
- Positional encoding recommended for temporal context
- Output: Enriched sequence `[B, T, D]`

### ✅ Multi-Head Output Layer (`MultiHeadOutput`)
- Predicts 4 token components independently:
  - `pitch` (128 classes)
  - `velocity_bin` (e.g. 32 bins)
  - `duration_bin` (e.g. 64 bins)
  - `time_bin` (e.g. 64 bins)
- Output: `{"pitch": [B, T, C], "velocity": ..., ...}`

## 🧮 Loss Function

### ✅ `compute_loss_classical`
- Uses `CrossEntropyLoss` for all components
- Supports:
  - Component weights (e.g., `pitch=1.0`, `velocity=0.3`)
  - Class weights (for imbalanced bins)

## 📊 Evaluation Metrics

### 🔢 Token-Level Accuracy
- `compute_accuracy`: Per-component top-1 accuracy
- `compute_note_token_accuracy`: All components must match

### 🎯 Streak-Based Consistency
- `compute_streak_score`: Rewards uninterrupted runs of correct predictions
- `compute_streak_compensated_score`: Offsets errors based on fluency

### 🎼 Optional Advanced Metrics
- `macro_f1_score_per_component`: For imbalanced class sets
- `flawless_bar_rate`: Counts how many bars/windows were perfectly predicted
- `error_heatmap`: Identifies component-wise model weaknesses

## 🧪 Hyperparameter Tuning Strategy

| Hyperparameter         | Best Metric To Tune Against     | Why                                  |
|------------------------|----------------------------------|--------------------------------------|
| Learning rate          | Total loss                      | Affects convergence                  |
| CNN kernel/stride      | `duration_acc`, `time_acc`      | Temporal structure sensitivity       |
| Transformer depth      | `note_token_acc`, `streak_score`| Long-term dependency modeling        |
| Loss weights           | `note_token_acc`, `streak_score`| Directly shapes learning focus       |
| Velocity bin size      | `velocity_acc`, F1              | Resolution of expression             |
| Duration bin size      | `duration_acc`, `macro_f1`      | Rhythmic resolution and coverage     |

## 🎮 Analogy: It's Guitar Hero

- Spectrogram scrolls like notes in Guitar Hero
- Model “plays” the correct note at the correct time
- Scored based on accuracy **and** consistency
- Musical phrasing matters more than isolated note correctness

## 🧠 Key Design Principles

- Treat token components independently
- Use modular metrics
- Reward consistency with streaks
- Penalize frequent small errors more than rare large ones
- Tune hyperparameters using *task-relevant* metrics
