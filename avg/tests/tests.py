import pandas as pd
from avg.main.models import Model, DecisionRuleModel, DecisionTree
from sklearn.model_selection import train_test_split

# df = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot_clean.csv')
df = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/latent_features_dataset.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
clf = Model(DecisionRuleModel)
# clf = Model(DecisionTree)

# X_train, X_test, y_train, y_test = clf.preprocess(X, y)  # For some reason causes the feature names in RIPPER rules to disappear

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

clf.train(X_train, y_train)
report = clf.report(X_test, y_test)
print(report.report)
print(f'Model size: {report.model_size}')
