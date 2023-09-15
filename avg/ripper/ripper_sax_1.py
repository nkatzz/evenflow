import argparse
import pandas as pd
from wittgenstein import RIPPER
from tslearn.piecewise import SymbolicAggregateApproximation
import re

def format_goal_name(goal):
    goal = goal.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return goal

def ripper_rules_to_prolog(ripper_rules):
    prolog_rules = []
    for goal, rule in ripper_rules:
        conditions = rule.conds
        prolog_conditions = ', '.join([f"{cond.feature}({cond.val})" for cond in conditions])
        prolog_rules.append(f"{format_goal_name(goal)} :- {prolog_conditions}.")
    return prolog_rules



"""
Call by

python decision_rules_sax_1.py /media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv
"""


def main(args):
    # Load the dataset
    data = pd.read_csv(args.data_path)

    # Define columns for features and labels
    feature_cols = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']
    label_col = 'goal_status'

    # SAX discretization
    sax = SymbolicAggregateApproximation(n_segments=10, alphabet_size_avg=10)

    # Prepare data for SAX discretization
    data_array = data[feature_cols].to_numpy().reshape(-1, len(feature_cols), 1)

    # Discretize using SAX
    sax_array = sax.fit_transform(data_array)

    # Flatten the 3D array to 2D to match the dataframe's shape
    sax_flattened = sax_array.reshape(sax_array.shape[0], -1)

    # Update the dataframe with the discretized data
    data[feature_cols] = sax_flattened

    # Split the data into training and testing sets
    train = data.sample(frac=0.7, random_state=42)
    test = data.drop(train.index)

    # One-vs-all RIPPER training and evaluation
    unique_classes = train[label_col].unique()
    classifiers = {}
    for cls in unique_classes:
        clf = RIPPER()
        y_train = (train[label_col] == cls).astype(int)
        y_test = (test[label_col] == cls).astype(int)
        
        clf.fit(train[feature_cols], y_train)
        score = clf.score(test[feature_cols], y_test)
        
        classifiers[cls] = clf
        print(f"RIPPER Test Accuracy for class {cls}: {score:.4f}")

    # Extract rules from the trained RIPPER models and print them
    all_ripper_rules = []
    for cls, clf in classifiers.items():
        ripper_rules = [(cls, rule) for rule in clf.ruleset_]
        all_ripper_rules.extend(ripper_rules)
        for _, rule in ripper_rules:
            print(cls, ":", rule)
        print("\n")



    # Convert all RIPPER rules to Prolog format and print
    prolog_rules = ripper_rules_to_prolog(all_ripper_rules)
    for rule in prolog_rules:
        print(rule)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RIPPER CLI for generating Prolog rules.")
    parser.add_argument('data_path', type=str, help="Path to the dataset CSV file.")
    args = parser.parse_args()

    main(args)

