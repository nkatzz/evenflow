import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import _tree
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Function to extract decision rules
def extract_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []
    path_ids = []

    def recurse(node, path, path_ids):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {threshold})"]
            recurse(tree_.children_left[node], p1, path_ids+[0])
            p2 += [f"({name} > {threshold})"]
            recurse(tree_.children_right[node], p2, path_ids+[1])
        else:
            path += [(tree_.value[node], tree_.weighted_n_node_samples[node])]
            paths.append(path)
            path_ids.append(path_ids)

    recurse(0, [], [])

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    return paths



data_path = '/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv'
df = pd.read_csv(data_path)

# Define the features to be discretized
features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']

# Instantiate the KBinsDiscretizer
kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

# Fit and transform the data
df[features] = kbd.fit_transform(df[features])

# Define the features and the target
X = df[features]
y = df['goal_status']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the size of the training set and the test set
X_train.shape, X_test.shape

# Instantiate the DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Set the size of the figure
plt.figure(figsize=(20,10))

# Plot the tree
"""
plot_tree(clf, 
          filled=True, 
          rounded=True, 
          class_names=clf.classes_, 
          feature_names=features, 
          max_depth=3) # limit depth for visualization purposes
plt.show()
"""

y_pred = clf.predict(X_test)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1-score: {f1}')

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Extract true positives, false positives, and false negatives
TP = np.diag(cm)  # True Positives are on the diagonal
FP = cm.sum(axis=0) - TP  # False Positives are column-wise sums
FN = cm.sum(axis=1) - TP  # False Negatives are row-wise sums

print(TP, FP, FN)

# Extract rules from the tree
rules = extract_rules(clf, features)

for r in rules:
    print(r)


# Get the class labels
class_labels = clf.classes_

# Display the class labels
print(class_labels)

# Function to get class label from a rule
def get_class_label(rule):
    return np.argmax(rule[-1][0])

# Group rules by class label
rules_by_class = {label: [] for label in class_labels}
for rule in rules:
    class_label = class_labels[get_class_label(rule)]
    rules_by_class[class_label].append(rule)

# Display the rules, grouped by class
for class_label, rules in rules_by_class.items():
    print(f"\nClass: {class_label}\n")
    for rule in rules:
        conditions = " & ".join(rule[:-1])
        print(f"{class_label} <-- {conditions}")
