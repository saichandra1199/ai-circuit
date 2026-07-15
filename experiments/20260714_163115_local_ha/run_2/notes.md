## Changes Made
- **Model Checkpoint**: Updated to use the best model from Run 1 to leverage previous learnings.
- **Training Epochs**: Set to 20 to ensure sufficient training time while monitoring for overfitting.
- **Optimizer Learning Rate**: Adjusted to 0.0001 to fine-tune the model's learning process.
- **Loss Focal Gamma**: Increased to 3.0 to better handle class imbalance by focusing more on hard-to-classify examples.
- **Random Erasing Augmentation**: Enabled to enhance model robustness by randomly occluding parts of the input images.

## Results Analysis
The macro F1 score of 0.7140 indicates a significant gap of -0.1860 from the target of 0.9, suggesting room for improvement. The per-class F1 scores reveal strengths and weaknesses:
- **Accessories**: High precision (1.0000) but low recall (0.6000) indicates the model is conservative, correctly identifying fewer instances.
- **Garment_Full_body**: Low performance (f1=0.4444) suggests the model struggles with this class, likely due to confusion with other garment types.
- **Garment_Lower_body**: Strong performance (f1=0.8000) shows effective classification.
- **Garment_Upper_body**: Moderate performance (f1=0.6667) indicates potential overlap with other classes.
- **Shoes**: Excellent performance (f1=0.9091) suggests the model effectively identifies this class.

Key insights include the need to address the low recall in Accessories and Garment_Full_body, which may be causing confusion with other classes.

## Further Improvements
1. **Class Rebalancing**: Implement techniques such as oversampling underrepresented classes or undersampling overrepresented ones to improve recall, especially for Accessories and Garment_Full_body.
2. **Enhanced Augmentations**: Experiment with additional augmentations (e.g., rotation, scaling) to improve model robustness and generalization across all classes.
3. **Fine-tuning Learning Rate**: Conduct a learning rate schedule or use learning rate finder techniques to identify an optimal learning rate that could enhance convergence and performance.