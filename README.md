# SpeechEnhancementDNN
This project uses a Deep Neural Network (DNN) with STFT-based features to enhance noisy speech by predicting clean spectral representations and reconstructing the signal via inverse STFT. Performance is evaluated using PESQ and STOI on data with varying SNR levels, showing improved speech quality and intelligibility.

## Overview
This project implements a Deep Neural Network (DNN)-based speech enhancement system to improve the quality and intelligibility of noisy speech signals. The system uses STFT to extract spectral features, applies a trained DNN model for enhancement, and reconstructs the enhanced speech using inverse STFT.

## Features
- Speech enhancement using Deep Learning (DNN)
- STFT-based feature extraction
- Noise reduction from speech signals
- Support for different SNR levels
- Objective evaluation using PESQ and STOI

## Dataset
The dataset includes:
- Clean speech audio
- Noise audio
- Mixed noisy speech generated at different SNR levels

## Methodology
1. Convert audio to frequency domain using STFT  
2. Extract spectral features  
3. Train DNN model on noisy vs clean mapping  
4. Predict enhanced speech features  
5. Reconstruct enhanced speech using inverse STFT  

## Evaluation Metrics
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)

## Results
The model improves speech clarity by reducing background noise and increasing intelligibility across different noise conditions.

## Requirements
- Python 3.x
- NumPy
- SciPy
- TensorFlow / PyTorch
- librosa
- soundfile

## How to Run
```bash
python train.py
python test.py
