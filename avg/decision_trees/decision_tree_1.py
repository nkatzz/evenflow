import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree
from scipy.spatial.transform import Rotation as R


# Function to extract rules from the decision tree
def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []
    path = []

    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 2)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 2)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.argmax(path[-1][0]))
        else:
            rule += "response: "+str(class_names[np.argmax(path[-1][0])])
        rule += " | based on "+ str(path[-1][1]) + " samples"
        rules += [rule]
        
    return rules


df = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')

# Select features and labels
features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']
label = 'goal_status'

# Convert categorical labels to numerical labels
le = LabelEncoder()
df[label] = le.fit_transform(df[label])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[label], test_size=0.3, random_state=0)

# Train the decision tree model
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Compute accuracy of the model on the test set
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f'Accuracy 1: {accuracy}')

# Plot the decision tree
"""
plt.figure(figsize=(15,10))
tree.plot_tree(clf, filled=True, feature_names=features, class_names=[str(i) for i in le.classes_])
plt.show()
"""

# Extract rules from the decision tree
rules = get_rules(clf, features, le.classes_)

for r in rules:
    print(r)


# Create 'velocity' feature
df['velocity'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['wz']**2)

# Convert quaternions to Euler angles
r = R.from_quat(df[['ox', 'oy', 'oz', 'ow']])
euler = r.as_euler('xyz', degrees=True)
df[['orient_x', 'orient_y', 'orient_z']] = euler

# Select new features
new_features = ['velocity', 'orient_x', 'orient_y', 'orient_z']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[new_features], df[label], test_size=0.3, random_state=0)

# Train the decision tree model
clf_new = DecisionTreeClassifier(random_state=0)
clf_new.fit(X_train, y_train)

# Make predictions on the test set
y_pred_new = clf_new.predict(X_test)

# Compute accuracy of the model on the test set
accuracy_new = accuracy_score(y_test, y_pred_new)

# Print the accuracy
print(f'Accuracy 2: {accuracy_new}')

rules_new = get_rules(clf_new, new_features, le.classes_)

for r in rules_new:
    print(r)
