import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Run the utils.pre_process_dataset first to get the clean dataset
df = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot_clean.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

unique_classes = y.unique()

# We'll scale our data to make the neural network's job a bit easier.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified Split: Ensuring that the train/test split has samples from all the classes
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0, stratify=y)


mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)

print('Finished training')

y_pred = mlp.predict(X_test)

# Display confusion matrix
confusion = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='g', xticklabels=unique_classes, yticklabels=unique_classes)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Display classification report, which includes F1-score for each class
report = classification_report(y_test, y_pred, digits=3, target_names=unique_classes)
print(report)

