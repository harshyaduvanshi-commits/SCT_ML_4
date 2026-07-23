# Hand Gesture Recognition using CNN

## Project Overview
This project builds a **Convolutional Neural Network (CNN)** to classify
10 different hand gestures from the Kaggle "LeapGestRecog" dataset -
near-infrared images captured by a Leap Motion sensor across 10 subjects
(~20,000 images total).

## Technologies Used
- Python
- OpenCV
- TensorFlow / Keras
- Scikit-Learn
- Matplotlib / Seaborn

## Workflow
1. Download the dataset and build a gesture label lookup dynamically
2. Load images and normalize pixel values
3. One-hot encode labels
4. Split data randomly (stratified across all 10 gesture classes)
5. Build a CNN with Batch Normalization, Spatial Dropout, and Global
   Average Pooling
6. Apply data augmentation (rotation, shift, zoom, brightness)
7. Train with Early Stopping to prevent overfitting
8. Evaluate using accuracy, a classification report, and a confusion matrix
9. Run 3-fold stratified cross-validation to check result consistency

## Dataset
- Source: Kaggle "LeapGestRecog" (gti-upm/leapgestrecog)
- 10 subjects, 10 gesture classes, ~20,000 images
- Gestures: palm, l, fist, fist_moved, thumb, index, ok, palm_moved, c, down

## Evaluation Metrics
- Accuracy
- Classification Report (precision, recall, f1-score per gesture)
- Confusion Matrix
- 3-Fold Cross-Validation (mean accuracy and standard deviation)

## Results
The model achieved approximately **99.9% test accuracy** using a random,
stratified train/test split (update with your actual numbers, including
the 3-fold cross-validation mean and standard deviation).

## Important Methodological Note
This version uses a **random, stratified split** across all subjects.
Because consecutive video frames of the same subject performing the same
gesture look nearly identical, this type of split allows near-duplicate
images to appear in both the training and test sets. This can inflate
accuracy compared to how the model would actually perform on a genuinely
new, unseen person - which is the real-world scenario implied by a
gesture-based control system.

A separate experiment using a **subject-based split** (holding out entire
subjects, e.g. 2 of the 10, so the model never sees their hands during
training) produced a more conservative test accuracy of approximately
**78%**. This is likely a more honest estimate of real-world performance
for a new user. A confusion matrix from that experiment showed the model
struggles most with visually similar gesture pairs (e.g. "fist" vs
"fist_moved"), which is a reasonable limitation given that a single
static frame cannot capture motion.

## Limitations
- The reported ~99.9% accuracy reflects performance on data from subjects
  already seen during training, not on new users
- Some gesture classes (e.g. fist vs fist_moved) are difficult to
  distinguish from single static frames, since the difference is
  primarily about motion over time
- Cross-validation folds are also randomly split across all subjects,
  so they share the same limitation as the main result

## Future Improvements
- Report both the random-split and subject-based-split accuracy side by
  side, rather than only the higher number
- Try video/sequence-based input (multiple consecutive frames) to better
  distinguish motion-based gestures like "fist_moved"
- Experiment with transfer learning using a pretrained image model
- Test generalization across more held-out subject combinations to check
  consistency


