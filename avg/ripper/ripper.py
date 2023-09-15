import pandas as pd
import wittgenstein as lw
from sklearn.model_selection import train_test_split

"""
This uses the RIPPER implementation in the wittgenstein package. RIPPER is designed for binary classification,
so we're using an one-against-all approach here, to learn decision rules for each class.
"""

# Load the dataset
dataset = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')

# Extract the specified columns as features and 'goal_status' as labels
features = dataset[['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']]
labels = dataset['goal_status']

# Split the dataset into training and test sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Get unique classes in goal_status
unique_classes = y_train.unique()

# Store classifiers and predictions
classifiers = {}
predictions = {}

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
    predictions[uc] = y_pred

    # Display accuracy for this class
    accuracy = (y_pred == y_test_binary).mean()
    print(f"Accuracy for {uc}: {accuracy:.4f}")

    # Display the learned rules
    print(ripper_clf.ruleset_)
    print("-" * 50)

