import pandas as pd
import numpy as np
import wittgenstein as lw
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')

# ------ SAX Transformation Implementation ------

def normalize_series(series):
    """Normalize the series to zero mean and unit variance."""
    return (series - series.mean()) / series.std()

def sax_transform_value(value, breakpoints):
    """Transform a single value using Symbolic Aggregate Approximation."""
    for i in range(len(breakpoints)):
        if value < breakpoints[i]:
            return chr(97 + i)
    return chr(97 + len(breakpoints))

def column_sax_transform(column, alphabet_size=10):
    """Transform an entire column (time series) using Symbolic Aggregate Approximation."""
    breakpoints = np.percentile(column, np.linspace(0, 100, alphabet_size+1)[1:-1])
    normalized_column = normalize_series(column)
    return [sax_transform_value(val, breakpoints) for val in normalized_column]


# ------ Apply SAX Transformation to Discretize the Dataset ------

discretized_df = dataset[['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']].apply(column_sax_transform)
discretized_df['goal_status'] = dataset['goal_status']


# ------ RIPPER Algorithm with One-Against-All Approach ------

# Split the discretized dataset into training and test sets (70/30 split)
X_train_discretized, X_test_discretized, y_train_discretized, y_test_discretized = train_test_split(
    discretized_df[['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']], 
    discretized_df['goal_status'], 
    test_size=0.3, 
    random_state=42
)

# Get unique classes in goal_status
unique_classes = y_train_discretized.unique()

# Store classifiers and predictions for the discretized dataset
classifiers_discretized = {}
predictions_discretized = {}

for uc in unique_classes:
    # Create binary labels
    y_train_binary = (y_train_discretized == uc).astype(int)
    y_test_binary = (y_test_discretized == uc).astype(int)
    
    # Initialize and train the RIPPER model
    ripper_clf_discretized = lw.RIPPER()
    ripper_clf_discretized.fit(X_train_discretized, y_train_binary, pos_class=1)
    
    # Predict on the test set
    y_pred_discretized = ripper_clf_discretized.predict(X_test_discretized)
    
    # Store the classifier and predictions
    classifiers_discretized[uc] = ripper_clf_discretized
    predictions_discretized[uc] = y_pred_discretized

    # Display accuracy for this class
    accuracy = (y_pred_discretized == y_test_binary).mean()
    print(f"Accuracy for {uc}: {accuracy:.4f}")

    # Display the learned rules
    print(ripper_clf_discretized.ruleset_)
    print("-" * 50)

