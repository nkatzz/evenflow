# Learn a decision tree

import pandas as pd
from sklearn import tree
from sklearn.tree import _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import graphviz

def get_rules(tree, feature_names, class_names):
    """Extract the rules from the tree"""
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
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    for p in paths:
        print(p) 
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (prob: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules



train = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')
class_names = train.goal_status.unique()

# Map class names to integers (not needed after all):
# class_dict = {name: index for index, name in enumerate(class_names)}
# train.replace({"goal_status": class_dict}, inplace=True)

# features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz', 'idle', 'linear', 'rotational']
features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']
# features = ['idle', 'linear', 'rotational']  # Crap F1-score (thank god...)
# features = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow']
# features = ['px', 'py', 'pz']  # We get an F1-score of 0.99 with this as well (but a much larger tree)
# features = ['vx', 'vy', 'wz']  # Macro F1-score: 0.884845377476306, Micro F1-score:0.975033988382153, huge tree


X = train.loc[:, features]
y = train.goal_status

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
actual = y_test
f1_macro = f1_score(actual, predicted, average='macro')
f1_micro = f1_score(actual, predicted, average='micro')
f1_none = f1_score(actual, predicted, average=None)

print(f'Macro F1-score: {f1_macro}\nMicro F1-score:{f1_micro}\nPer class:{f1_none}')

# Visualize the tree
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=features, 
                                class_names=class_names, filled=True, rounded=True, 
                                special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.view()

text_representation = tree.export_text(clf, feature_names=features)
print(f'\nThe tree is:\n{text_representation}')
rules = '\n'.join(get_rules(clf, features, class_names))
print(f"""\nThe rules are:\n{rules}""")


