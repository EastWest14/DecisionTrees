import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import graphviz
from sklearn.metrics import roc_auc_score


df = pd.read_csv('GermanCredit.csv')
for c in df.columns:
	print(c)
	print(df[c].describe())
	print('===')

# feature_names = ['Property.RealEstate', 'Duration']
feature_names = ["Duration","Amount","InstallmentRatePercentage","ResidenceDuration","Age","NumberExistingCredits","NumberPeopleMaintenance","Telephone","ForeignWorker","CheckingAccountStatus.lt.0","CheckingAccountStatus.0.to.200","CheckingAccountStatus.gt.200","CheckingAccountStatus.none","CreditHistory.NoCredit.AllPaid","CreditHistory.ThisBank.AllPaid","CreditHistory.PaidDuly","CreditHistory.Delay","CreditHistory.Critical","Purpose.NewCar","Purpose.UsedCar","Purpose.Furniture.Equipment","Purpose.Radio.Television","Purpose.DomesticAppliance","Purpose.Repairs","Purpose.Education","Purpose.Vacation","Purpose.Retraining","Purpose.Business","Purpose.Other","SavingsAccountBonds.lt.100","SavingsAccountBonds.100.to.500","SavingsAccountBonds.500.to.1000","SavingsAccountBonds.gt.1000","SavingsAccountBonds.Unknown","EmploymentDuration.lt.1","EmploymentDuration.1.to.4","EmploymentDuration.4.to.7","EmploymentDuration.gt.7","EmploymentDuration.Unemployed","Personal.Male.Divorced.Seperated","Personal.Female.NotSingle","Personal.Male.Single","Personal.Male.Married.Widowed","Personal.Female.Single","OtherDebtorsGuarantors.None","OtherDebtorsGuarantors.CoApplicant","OtherDebtorsGuarantors.Guarantor","Property.RealEstate","Property.Insurance","Property.CarOther","Property.Unknown","OtherInstallmentPlans.Bank","OtherInstallmentPlans.Stores","OtherInstallmentPlans.None","Housing.Rent","Housing.Own","Housing.ForFree","Job.UnemployedUnskilled","Job.UnskilledResident","Job.SkilledEmployee","Job.Management.SelfEmp.HighlyQualified"]

X = df[feature_names]
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

param_grid = {
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3],
}

# Dataset is not fully balanced, will use AOC curve score instead of accuracy.
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc', cv=5)

grid_search.fit(X_train, y_train)

print(f"Grid search results: {grid_search.cv_results_}")
best_params = grid_search.best_params_

# Print the best parameters and best score
print("Best Parameters:", best_params)
print("Best Score:", grid_search.best_score_)

clf = DecisionTreeClassifier(
	max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
)

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# print(y_pred)

# # Generate the classification report
report = classification_report(y_test, y_pred)

auc_score = roc_auc_score(y_test, y_pred)
print("AUC Score:", auc_score)

# # Print the classification report
# print(report)

# # print(tree.export_graphviz(clf, out_file=None))
# dot_data = tree.export_graphviz(clf, feature_names=feature_names, filled=True, rounded=True, out_file=None)

# # Create a graph from the DOT data
# graph = graphviz.Source(dot_data)

# # Display the decision tree
# graph.view()
# graph.render()


# colormap = {0: 'red', 1: 'green'}

# # Create the scatter plot
# plt.scatter(df['Duration'], df['Property.RealEstate'], c=[colormap[label] for label in y])

# # Set labels and title
# plt.xlabel('Duration')
# plt.ylabel('Property.RealEstate')
# plt.title('Data Visualization')


# plt.legend()

# Show the plot
# plt.show()

