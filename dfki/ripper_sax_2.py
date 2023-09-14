import pandas as pd
from tslearn.piecewise import SymbolicAggregateApproximation
from wittgenstein import RIPPER
from collections import defaultdict

# Load dataset
data = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')

# Feature columns for SAX transformation and RIPPER
feature_cols = ['px', 'py', 'pz', 'ox', 'oy', 'oz', 'ow', 'vx', 'vy', 'wz']

# SAX discretization
sax = SymbolicAggregateApproximation(n_segments=10, alphabet_size_avg=10)
data_array = data[feature_cols].to_numpy().reshape(-1, len(feature_cols), 1)
sax_array = sax.fit_transform(data_array)
data[feature_cols] = sax_array.reshape(-1, len(feature_cols))

# Split data into training and testing sets
train_data = data.sample(frac=0.7)
test_data = data.drop(train_data.index)

# Helper function to format the goal name for Prolog
def format_goal_name(goal):
    return goal.lower().replace(" ", "_").replace("(", "").replace(")", "")

# Convert RIPPER rules to Prolog format
def ripper_rules_to_prolog(ripper_rules):
    prolog_rules = []
    for goal, rule in ripper_rules:
        conditions = rule.conds
        prolog_conditions = ', '.join([f"{cond.feature}({cond.val})" for cond in conditions])
        prolog_rules.append(f"{format_goal_name(goal)} :- {prolog_conditions}.")
    return prolog_rules

# Train RIPPER in one-vs-all setting
all_ripper_rules = []
for goal in data['goal_status'].unique():
    clf = RIPPER()
    y = (train_data['goal_status'] == goal).astype(int)
    clf.fit(train_data[feature_cols], y)
    ripper_rules = [(goal, rule) for rule in clf.ruleset_]
    all_ripper_rules.extend(ripper_rules)

print("\nRIPPER output:\n")
for goal, rule in all_ripper_rules:
    print(goal, ":", rule)

prolog_rules = ripper_rules_to_prolog(all_ripper_rules)
print("\nAs LP rules:\n")
for rule in prolog_rules:
    print(rule)

# Abstract rules using symbols e_1, e_2, ...
condition_to_symbol = defaultdict(lambda: f'e_{len(condition_to_symbol) + 1}')
abstract_rules = []
for goal, rule in all_ripper_rules:
    conditions = rule.conds
    abstract_conditions = [condition_to_symbol[str(cond)] for cond in conditions]
    abstract_rules.append(f"{format_goal_name(goal)} :- {', '.join(abstract_conditions)}.")

print("\nAbstract rules:\n")
for rule in abstract_rules:
    print(rule)

print("\nMapping from symbols to conditions:\n")
for symbol, condition in condition_to_symbol.items():
    print(f"{condition} : {symbol}")

