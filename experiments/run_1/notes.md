## Changes Made
This is the baseline run — the initial configuration includes standard data preprocessing, a simple CNN architecture, and a learning rate of 0.001. The dataset was split into training and validation sets with no augmentation applied.

## Results Analysis
The macro F1 score of 0.2485 indicates significant room for improvement, falling short of the target of 0.75 by a large margin. The accuracy of 0.2413 further highlights the model's inability to generalize well. Per-class F1 scores reveal that "Shoes" performed the best (f1=0.4423), suggesting that the model can identify this class with reasonable precision and recall. However, all other classes, particularly "Garment Lower body" (f1=0.1746) and "Accessories" (f1=0.1818), performed poorly, indicating potential issues with class imbalance or feature representation. The confusion matrix likely shows high misclassification rates among the lower-performing classes, which could be contributing to the overall low macro F1 score.

## Further Improvements
1. Implement data augmentation techniques to enhance the diversity of the training dataset, potentially improving the model's robustness and generalization.
2. Experiment with a more complex model architecture, such as a pre-trained backbone (e.g., ResNet or EfficientNet), to leverage transfer learning and improve feature extraction.
3. Address class imbalance by employing techniques such as oversampling minority classes or using class weights during training to ensure the model pays more attention to underrepresented classes.