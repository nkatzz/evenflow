from models import Model, DecisionTree, DecisionRuleModel
import pandas as pd
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":

    df = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot_clean.csv')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # clf = Model(DecisionTree)
    # clf = Model(DecisionTree, discretize='sax', bins=40, max_depth=10, criterion='entropy')
    clf = Model(DecisionRuleModel)
    # clf = Model(MLPClassifier)

    X_train, X_test, y_train, y_test = clf.preprocess(X, y)  # Also performs train/test splits

    clf.train(X_train, y_train)

    report = clf.report(X_test, y_test)
    if report.tree is not None:
        print(report.tree + '\n')
    if report.rules is not None and isinstance(clf, DecisionTree):
        print('\n'.join(report.rules))
    print("Performance on testing set:")
    print(report.report)
    if report.model_size is not None:
        print(f'Model size: {report.model_size}')



