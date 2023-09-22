from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import wittgenstein as lw
from sklearn.tree import _tree
from sklearn import tree
import numpy as np
import graphviz
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from avg.utils.utils import normalize_series, sax_transform_value, column_sax_transform
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


class Model:
    def __init__(self, model, data_format='dataframe', discretize=None, bins=0, **kwargs):
        """
        A common interface for various models.
        :param model: Name of a model (e.g. MLPClassifier, DecisionTree, SimpleVAE, Autoencoder...)
        :data_format: 'dataframe' or 'tensor'. Convert the data in the proper format for different models.
        :discretize: discretization method (None, sax, kbins)
        :bins: Number of bins/alphabet size for discretization
        :param kwargs: Hyperparameters for the chosen classifier

        Example usage:
        clf = Model(DecisionTree, discretize=sax, bins=10, max_depth=10, criterion='entropy')
        X_train, X_test, y_train, y_test = clf.preprocess(X, y)  # X, y are data and labels dataframes
        clf.train(X_train, y_train)
        accuracy = clf.evaluate(X_test, y_test)

        Instantiation examples:

        """

        self.model = model(**kwargs)
        self.data_format = data_format
        self.discretize = discretize
        self.bins = bins
        self.feature_names = []

    def preprocess(self, X, y):

        self.feature_names = X.columns.tolist()

        if (isinstance(self.model, DecisionTree) or isinstance(self.model,
                                                               DecisionRuleModel)) and self.discretize is not None:

            if self.discretize == 'sax':
                X = X.apply(column_sax_transform, alphabet_size=self.bins)

                # The above will generate string symbols, which won't work with sklearn's decision trees.
                # Convert the symbols to integers if out model is a tree
                if isinstance(self.model, DecisionTree):
                    # Convert SAX symbols to integer values
                    unique_symbols = np.unique(X.values)
                    for i, symbol in enumerate(unique_symbols):
                        X[X == symbol] = i

            else:  # k-bins discretization (a-temporal)
                discretizer = KBinsDiscretizer(n_bins=self.bins, encode='ordinal', strategy='uniform')
                X = discretizer.fit_transform(X)

        # Neural classifiers
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        if self.data_format == 'tensor':
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y.values, dtype=torch.float32) if isinstance(y, pd.Series) else torch.tensor(y,
                                                                                                          dtype=torch.float32)

        # Split the data in a stratified manner
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Returns an accuracy score
        """
        return self.model.score(X_test, y_test)

    def report(self, X_test, y_test):
        """
        Generates a classification report on test data. For trees and rules it also returns the learnt model.
        Includes F1-scores per class. This only works for supervised learning models (i.e., not autoencoders)
        """

        if isinstance(self.model, DecisionRuleModel):
            report = self.model.report(X_test, y_test)
            return report
        else:
            y_pred = self.model.predict(X_test)
            unique_classes = y_test.unique()
            report = classification_report(y_test, y_pred, digits=3, target_names=unique_classes, zero_division=0.0)

            if isinstance(self.model, DecisionTree):
                feature_names = self.feature_names if self.feature_names else X_test.columns.tolist()
                _rules_ = self.model.get_rules(feature_names)
                _tree_ = self.model.get_tree(feature_names)
                return Report(report, rules=_rules_, tree=_tree_)
            elif isinstance(self.model, MLPClassifier):
                return Report(report)
            else:
                print("""This shouldn't have happened""")

    def predict(self, data):
        return self.model.predict(data)


class Report:
    def __init__(self, report, rules=None, tree=None, model_size=None, macro_f1_train=None):
        self.report = report
        self.rules = rules
        self.tree = tree
        self.model_size = model_size
        self.macro_f1_train = macro_f1_train


class DecisionRuleModel:
    def __init__(self, multiclass=True):
        self.multiclass = multiclass
        self.classifiers = {}  # different RIPPER instances are trained for each class
        self.f1_train_set = {}  # F1-scores per class (key)
        self.f1_test_set = {}  # same
        self.predictions_test_set = []
        self.training_set_performance = 0.0
        self.model_size = 0

    def fit(self, X_train, y=None):

        unique_classes = y.unique()

        for uc in unique_classes:
            print(f"Class: {uc}")

            ripper_clf = lw.RIPPER(verbosity=1)  # prune_size=20
            ripper_clf.fit(X_train, y, pos_class=uc)

            # Predict on the training set
            y_pred = ripper_clf.predict(X_train)

            self.classifiers[uc] = ripper_clf

            # Create binary labels. This is needed for the f1_score method to work.
            y_train_binary = (y == uc).astype(int)
            f1 = f1_score(y_train_binary, y_pred)
            self.f1_train_set[uc] = f1

            model_size = sum([len(rule) for rule in ripper_clf.ruleset_])
            self.model_size += model_size

            print(ripper_clf.ruleset_)
            print(f'Training set F1: {f1}')
            print("-" * 50)

    def report(self, X_test, y_test):

        unique_classes = y_test.unique()
        all_predictions = []
        for uc in unique_classes:
            clf = self.classifiers[uc]
            y_pred = clf.predict(X_test)
            all_predictions.append(y_pred)

        # Convert list of predictions to an array
        all_predictions = np.array(all_predictions).T

        # Convert the 2D array of predictions into a single 1D array of labels
        predicted_labels = [unique_classes[row] for row in np.argmax(all_predictions, axis=1)]

        report = classification_report(y_test, predicted_labels, digits=3, zero_division=0.0)
        r = Report(report, model_size=self.model_size)
        return r


class DecisionTree(DecisionTreeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __extract_rules(self, feature_names):
        tree_ = self.tree_
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
                recurse(tree_.children_left[node], p1, path_ids + [0])
                p2 += [f"({name} > {threshold})"]
                recurse(tree_.children_right[node], p2, path_ids + [1])
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

    def get_rules(self, feature_names):
        """
        Return the rule in a more print-friendly format
        """

        def get_class_label(rule):
            return np.argmax(rule[-1][0])

        class_labels = self.classes_

        rules = self.__extract_rules(feature_names)

        # Group rules by class label
        rules_by_class = {label: [] for label in class_labels}
        for rule in rules:
            class_label = class_labels[get_class_label(rule)]
            rules_by_class[class_label].append(rule)

        rules_out = []
        for class_label, rules in rules_by_class.items():
            for rule in rules:
                conditions = " & ".join(rule[:-1])
                rules_out.append(f"{class_label} <-- {conditions}")

        return rules_out

    def get_tree(self, features):
        """
        Return a text representation of the tree.
        """
        text_representation = tree.export_text(self, feature_names=features)
        return text_representation

    def visualize(self, features, class_names):
        """
        Visualize the tree.
        """
        dot_data = tree.export_graphviz(self, out_file=None, feature_names=features,
                                        class_names=class_names, filled=True, rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.view()


class SAX:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size

    # ------ SAX Transformation Implementation ------
    @staticmethod
    def normalize_series(series):
        """Normalize the series to zero mean and unit variance."""
        return (series - series.mean()) / series.std()

    @staticmethod
    def sax_transform_value(value, breakpoints):
        """Transform a single value using Symbolic Aggregate Approximation."""
        for i in range(len(breakpoints)):
            if value < breakpoints[i]:
                return chr(97 + i)
        return chr(97 + len(breakpoints))

    def column_sax_transform(self, column):
        """Transform an entire column (time series) using Symbolic Aggregate Approximation."""
        breakpoints = np.percentile(column, np.linspace(0, 100, self.alphabet_size + 1)[1:-1])
        normalized_column = self.normalize_series(column)
        return [self.sax_transform_value(val, breakpoints) for val in normalized_column]
