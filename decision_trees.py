from typing import List

import graphviz
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

FILE_NAME = "GermanCredit.csv"
LABEL_NAME = "Class"
LABEL_MAPPING = {"Good": 1, "Bad": 0}
# Seed for pseudo-random splitting of data into test and training sets:
DATA_SPLIT_SEED = 20
DATA_SPLIT_TEST_FRACTION = 0.3


# Decision Tree Params:
DECISION_TREE_PARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 3, 5],
}
NUM_CROSS_VALIDATION_FOLDS_DT = 5


# Logistic Regression Params:
MAX_NUM_ITERATIONS_LR = 500


# Random Forest Params:
RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [30, 45, 60],
    "max_depth": [5, 8, 10],
    "min_samples_split": [3, 5, 8],
    "min_samples_leaf": [1, 2, 3],
}
NUM_CROSS_VALIDATION_FOLDS_RF = 5


# PDP Plotting Params:
PDP_PLOTTING_PARAMS = {
    "subsample": 50,
    "n_jobs": 2,
    "grid_resolution": 20,
    "random_state": 0,
}
# Pick no more than 5.
FEATURES_TO_PLOT = [
    "Duration",
    "Amount",
    "InstallmentRatePercentage",
    "Job.UnskilledResident",
]

DEBUG_OUTPUT = False


def load_data(file_name: str, describe_columns: bool = False) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    if describe_columns:
        for c in df.columns:
            print(f"Column Name: {c}")
            print(df[c].describe())
            print("===")
    return df


def generate_feature_names(debug_output: bool = False) -> List[str]:
    feature_names = list(df.columns)
    feature_names.remove(LABEL_NAME)
    if debug_output:
        print(feature_names)
    return feature_names


def print_separator():
    print("")
    print("===")
    print("")


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    debug_output: bool = False,
) -> float:
    """Trains decision tree to classify the data into 'Good' and 'Bad' credit classes.
    Outputs area under curve (AUC) on the test data"""
    clf = DecisionTreeClassifier()

    # Dataset is not fully balanced, will use AOC curve score instead of accuracy.
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=DECISION_TREE_PARAM_GRID,
        scoring="roc_auc",
        cv=NUM_CROSS_VALIDATION_FOLDS_DT,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    if debug_output:
        print(f"Best parameters for decision trees: {best_params}")
        print(
            f"Best score for decision tree (on training data): {grid_search.best_score_}"
        )

    clf = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
    )
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    if debug_output:
        print(report)

    auc_score_decision_tree = roc_auc_score(y_test, y_pred)

    # Using Graphviz to plot the tree. Open Source.gv.pdf file to see rendered tree.
    dot_data = tree.export_graphviz(
        clf, feature_names=feature_names, filled=True, rounded=True, out_file=None
    )
    graph = graphviz.Source(dot_data)
    graph.view()
    graph.render()

    return auc_score_decision_tree


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    debug_output: bool = False,
) -> float:
    clf = LogisticRegression(max_iter=MAX_NUM_ITERATIONS_LR)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    if debug_output:
        print(report)

    auc_score = roc_auc_score(y_test, y_pred)
    return auc_score


def plot_pdp_on_features(clf: ClassifierMixin, X_train: pd.DataFrame):
    features_info = {
        "features": FEATURES_TO_PLOT,
        "kind": "average",
    }
    fig, ax = plt.subplots(ncols=1, nrows=len(FEATURES_TO_PLOT))
    fig.subplots_adjust(hspace=1.0)
    display = PartialDependenceDisplay.from_estimator(
        clf,
        X_train,
        **features_info,
        ax=ax,
        **PDP_PLOTTING_PARAMS,
    )
    _ = display.figure_.suptitle(
        "Partial Dependency Plot\n",
        fontsize=6,
    )
    plt.show()


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    debug_output: bool = False,
) -> float:
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=RANDOM_FOREST_PARAM_GRID,
        scoring="roc_auc",
        cv=NUM_CROSS_VALIDATION_FOLDS_RF,
    )

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    if debug_output:
        print(f"Best Parameters for forest: {best_params}")
        print(f"Best Score for forest: {grid_search.best_score_}")

    clf = RandomForestClassifier(
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        min_samples_split=best_params["min_samples_split"],
        n_estimators=best_params["n_estimators"],
    )
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    if debug_output:
        print(report)

    auc_score = roc_auc_score(y_test, y_pred)
    plot_pdp_on_features(clf, X_train)
    return auc_score


df = load_data(FILE_NAME, describe_columns=DEBUG_OUTPUT)
feature_names = generate_feature_names(debug_output=DEBUG_OUTPUT)

X = df[feature_names]
y = df["Class"].map(LABEL_MAPPING)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=DATA_SPLIT_TEST_FRACTION,
    random_state=20,
)

if DEBUG_OUTPUT:
    print(f"Number of training records: {X_train.shape[0]}")
    print(f"Number of test records: {X_test.shape[0]}")

print_separator()
decision_tree_auc_score = train_decision_tree(
    X_train, y_train, X_test, y_test, debug_output=DEBUG_OUTPUT
)
print(f"AUC Score Decision Tree: {decision_tree_auc_score}")

print_separator()

auc_score_logistic_regression = train_logistic_regression(
    X_train, y_train, X_test, y_test, debug_output=DEBUG_OUTPUT
)
print(f"AUC Score Logistic Regression: {auc_score_logistic_regression}")

print_separator()
auc_score_random_forest = train_random_forest(
    X_train, y_train, X_test, y_test, debug_output=DEBUG_OUTPUT
)
print(f"AUC Score Random Forest: {auc_score_random_forest}")
