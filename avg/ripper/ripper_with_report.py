import pandas as pd
import wittgenstein as lw
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

dataset = pd.read_csv('/home/nkatz/dev/evenflow/avg/vq_vae_triplet_loss/latent_representation_multi.csv')

features = dataset.iloc[:, :-1]
labels = dataset.iloc[:, -1]

# Split the dataset into training and test sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Get unique classes in goal_status
unique_classes = y_train.unique()

# Store classifiers and predictions
classifiers = {}
all_predictions = []

for uc in unique_classes:
    print(f"Training for class: {uc}")

    # Create binary labels
    y_train_binary = (y_train == uc).astype(int)
    y_test_binary = (y_test == uc).astype(int)

    # Initialize and train the RIPPER model
    ripper_clf = lw.RIPPER()
    ripper_clf.fit(X_train, y_train_binary, pos_class=1)

    # Predict on the test set
    y_pred = ripper_clf.predict(X_test)

    # Store the classifier and predictions
    classifiers[uc] = ripper_clf
    all_predictions.append(y_pred)

    # Display accuracy for this class
    accuracy = (y_pred == y_test_binary).mean()
    print(f"Accuracy for {uc}: {accuracy:.4f}")

    conf_matrix = confusion_matrix(y_test_binary, y_pred)
    TPs = conf_matrix[1, 1]
    FPs = conf_matrix[0, 1]
    FNs = conf_matrix[1, 0]

    print(conf_matrix)
    print(f'F1-score: {f1_score(y_test_binary, y_pred)} (TPs, FPs, FNs: {TPs, FPs, FNs})')

    # Display the learned rules
    print(ripper_clf.ruleset_)
    print("-" * 50)

# Convert list of predictions to an array
all_predictions = np.array(all_predictions).T

# Convert the 2D array of predictions into a single 1D array of labels
predicted_labels = [unique_classes[row] for row in np.argmax(all_predictions, axis=1)]

# Print classification report
report = classification_report(y_test, predicted_labels, labels=unique_classes, digits=3, zero_division=0.0)
print(report)
