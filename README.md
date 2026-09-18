# Flower Classifier with Vision Transformer 

**Project Goal**: Iteratively improve a transfer learning pipeline from about 85% to 96% accuracy using techniques from *Hands-on Machine Learning from PyTorch (Chapter 12)*

## Results Summary

| **Notebook** | **Strategy**                | **Val Accuracy** |
|--------------|-----------------------------|------------------|
| 02           | Head-only (Frozen)          |                  |
| 03           | Full fine-tune              |                  |
| 04           | Differential LR + Scheduler |                  |
| 05           | Ensemble(ViT + ResNet)      |                  |

## Key Learnings
- **Differential Learning Rates**: Applying a 100x similar LR to lower layers prevented catastrophic forgetting.
- **Failure Analysis**: The model struggled most with yellow flowers (daisy vs dandelion). I adjusted augmentation (ColorJitter) to address this.
- **Ensembling**: Averaging ViT and ResNet logits smoothed out individual model biases.

