import numpy as np

# import sklearn classifiers
from sklearn.ensemble import AdaBoostClassifier  # boosting with decision trees as default base model
from sklearn.naive_bayes import CategoricalNB  # categorical naive bayes
from sklearn.linear_model import LogisticRegressionCV  # categorical logistic regression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# import sklearn utilities
from sklearn.model_selection import train_test_split  # for paritioning data into disjoint training and testing subsets
from sklearn.preprocessing import OneHotEncoder  # for converting string features to numeric vectors
# import catboost classifier for categorical data
from catboost import CatBoostClassifier


from database import parsed_files, extract_clustering_features, clustering_feature_labels, query_rows
import matplotlib.pyplot as plt

def extract_features_and_labels(rows_dict):
    """This function takes the rows of a parsed team matchup csv
    file and extracts the features and labels for the classification
    problem of predicting which team wins. Does not convert the features
    to numeric values."""
    features = []
    labels = []
    for line in rows_dict:
        visitor_name = line["Visitor/Neutral"]
        visitor_points = int(line["PTS_Visitor"])  # make sure to convert to a number for numeric comparison
        home_name = line["Home/Neutral"]
        home_points = int(line["PTS_Home"])  # make sure to convert to a number for numeric comparison
        features.append([home_name, visitor_name])
        labels.append(1 if home_points >= visitor_points else -1)  # 1 means home win or draw, -1 means visitor win
    return features, labels

# get the string features and numeric labels
matchups_data = parsed_files["team_matchups.txt"]
features, labels = extract_features_and_labels(matchups_data)

# encode categorical string features to numbers
encoder = OneHotEncoder()
features = encoder.fit_transform(features).toarray()  # fit the data and transform it

# split to training and testing data
train_fraction = 0.8            # fraction of input to use for training (rest is used for testing)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=train_fraction)

# create the classifiers
bayes = CategoricalNB()
logistic_regression = LogisticRegressionCV()
adaboost = AdaBoostClassifier()
k_choice = 3
knn = KNeighborsClassifier(n_neighbors=k_choice, metric="hamming")
svm = SVC(C=1.0, kernel='rbf')
neural_network = MLPClassifier(hidden_layer_sizes=(300, 100))
catboost = CatBoostClassifier(iterations=1500, loss_function="Logloss", verbose=False, depth=2)

model_names, models = ["Naive Bayes", "Logistic Regression", "AdaBoost", f"K Nearest Neighbours, k={k_choice}", "Support Vector Machine", "Neural Network", "CatBoost"], [bayes, logistic_regression, adaboost, knn, svm, neural_network, catboost]

# train the classifiers
for model in models:
    model.fit(train_features, train_labels)

# score on train and test data
for model_name, model in zip(model_names, models):
    print(f"{model_name} train/test scores: {model.score(train_features, train_labels)}/{model.score(test_features, test_labels)}")
