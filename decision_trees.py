import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('GermanCredit.csv')

X = df[['Property.RealEstate']]
print(X)

label_mapping = {'Good': 1, 'Bad': 0}
y = df['Class'].map(label_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

# Print the shapes of the resulting arrays
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print(tree.export_graphviz(clf, out_file=None))