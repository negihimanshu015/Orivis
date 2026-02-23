# Orivis Dataset Strategy

To ensure a robust and generalized synthetic media detection platform, Orivis will utilize a diverse set of datasets and rigorous validation protocols.

## 1. Selected Datasets

### Video Detection
- **FaceForensics++ (FF++)**: 
    - **Description**: Benchmarking standard with 1,000 original videos and 4,000 manipulated ones (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
    - **Use Case**: Primary training for high-quality artifact detection.
- **DeepFake Detection Challenge (DFDC) Preview/Full**:
    - **Description**: Large-scale dataset (over 100k videos) with varied conditions (lighting, distance, resolution).
    - **Use Case**: Training for robustness and real-world variability.

### Audio Detection
- **ASVspoof 2019/2021**:
    - **Description**: Standard dataset for automatic speaker verification spoofing and deepfake detection.
    - **Use Case**: Primary training for audio artifacts (Logical Access and Physical Access).

## 2. Partitioning Strategy

We will adopt a **80/10/10** split across all datasets to ensure consistent evaluation.

| Split | Description | Purpose |
| :--- | :--- | :--- |
| **Train (80%)** | Combined samples from all primary datasets. | Model parameter optimization. |
| **Val (10%)** | Hold-out set from training datasets. | Hyperparameter tuning and checkpoint selection. |
| **Test (10%)** | Final evaluation set from training datasets. | Benchmarking performance on seen distributions. |

## 3. Cross-Dataset Validation (Generalization)

To ensure the model isn't just "memorizing" specific dataset artifacts:
- **Protocol**: Train on FF++ and evaluate on DFDC (and vice versa).
- **Out-of-Distribution (OOD)**: Evaluate on unseen datasets (e.g., Celeb-DF) to test performance against new generation techniques.

## 4. Preprocessing & Augmentation
- **Video**: Key-frame extraction (10-20 frames per video), face cropping/alignment.
- **Audio**: Conversion to Log-Mel Spectrograms, trimming/padding to fixed length.
- **Augmentation**: Noise injection, compression (various bitrates), resizing, and blur.
