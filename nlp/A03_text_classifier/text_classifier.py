import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from find_best_hyperparameters import fit_tuning
from find_best_hyperparameters import get_multinomial_naive_bayes_params
from find_best_hyperparameters import get_support_vector_params
from find_best_hyperparameters import get_logistic_regression_params
from find_best_hyperparameters import get_tfidf_params
from find_best_hyperparameters import read_dataset


# Accuracy:
# MNB: 0.611
# LR: 0.711
# SVC: 0.844
def classifiers():
    df, X, y = read_dataset("./NSF.csv")

    print("\nDataset content:")
    print(df.head(5))

    print("\nClasses couting:")
    print(df["class"].value_counts())

    print(f"\nDataset size: {len(df)}")

    print("\nTrainig and Test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=1979)
    print(f"len(X_train): {len(X_train)}")
    print(f"len(X_test): {len(X_test)}")

    # Multinomial Naive Bayes classifier inicialization
    pipeMNB = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])

    # Logistic regression inicialization
    pipeLR  = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(random_state=1979))])

    # Support Vector Classifier (SVC) inicialization    
    pipeSVC = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])

    print("\nAccuracy:", end="")
    pipeMNB.fit(X_train, y_train)
    predictMNB = pipeMNB.predict(X_test)
    print(f"\nMNB: {accuracy_score(y_test, predictMNB):.3f}")
    pipeLR.fit(X_train, y_train)
    predictLR = pipeLR.predict(X_test)
    print(f"LR: {accuracy_score(y_test, predictLR):.3f}")
    pipeSVC.fit(X_train, y_train)
    predictSVC = pipeSVC.predict(X_test)
    print(f"SVC: {accuracy_score(y_test, predictSVC):.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictSVC))


# Accuracy:
# MNB: 0.811
# LR: 0.833
# SVC: 0.844
def classifiers_tuning():
    df, X, y = read_dataset("./NSF.csv")

    print("\nDataset content:")
    print(df.head(5))

    print("\nClasses couting:")
    print(df["class"].value_counts())

    print(f"\nDataset size: {len(df)}")

    print("\nTrainig and Test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=1979)
    print(f"len(X_train): {len(X_train)}")
    print(f"len(X_test): {len(X_test)}")

    # Multinomial Naive Bayes classifier inicialization
    param_grid_MNB = {'alpha': np.linspace(0.1, 1.5, 10), 'fit_prior': [True, False], 'force_alpha': [True, False]}
    gridCV_MNB = GridSearchCV(MultinomialNB(), param_grid_MNB, cv=5, scoring="f1_macro")
    pipeMNB = Pipeline([("tfidf", TfidfVectorizer()), ("clf", gridCV_MNB)])

    # Logistic regression inicialization
    param_grid_LR = {'C': np.logspace(-5, 8, 15)} 
    gridCV_LR = GridSearchCV(LogisticRegression(random_state=1979), param_grid_LR, cv=5, scoring="f1_macro")
    pipeLR  = Pipeline([("tfidf", TfidfVectorizer()), ("clf", gridCV_LR)])

    # Support Vector Classifier (SVC) inicialization    
    param_grid_SVC = {"class_weight": [None, "balanced"], "tol": np.logspace(-5, 8, 15)} 
    gridCV_SVC = GridSearchCV(LinearSVC(), param_grid_SVC, cv=5, scoring="f1_macro")
    pipeSVC = Pipeline([("tfidf", TfidfVectorizer()), ("clf", gridCV_SVC)])

    print("\nAccuracy:", end="")
    mnb = pipeMNB.fit(X_train, y_train)
    predictMNB = pipeMNB.predict(X_test)
    print(f"\nMNB: {accuracy_score(y_test, predictMNB):.3f}")
    lr = pipeLR.fit(X_train, y_train)
    predictLR = pipeLR.predict(X_test)
    print(f"LR: {accuracy_score(y_test, predictLR):.3f}")
    svc = pipeSVC.fit(X_train, y_train)
    predictSVC = pipeSVC.predict(X_test)
    print(f"SVC: {accuracy_score(y_test, predictSVC):.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictSVC))

    print("\nMNB best params:")
    print("  Best Score: ", gridCV_MNB.best_score_)
    print("  Best Params: ", gridCV_MNB.best_params_)
    print("LR best params:")
    print("  Best Score: ", gridCV_LR.best_score_)
    print("  Best Params: ", gridCV_LR.best_params_)
    print("SVC best params:")
    print("  Best Score: ", gridCV_SVC.best_score_)
    print("  Best Params: ", gridCV_SVC.best_params_)

    print("\nCV Results:")
    print(pd.DataFrame(gridCV_MNB.cv_results_))


def pipeline_tuning():
    df, X, y = read_dataset("./NSF.csv")

    print("\nDataset content:")
    print(df.head(5))

    print("\nClasses couting:")
    print(df["class"].value_counts())

    print(f"\nDataset size: {len(df)}")

    print("\nTrainig and Test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=1979)
    print(f"len(X_train): {len(X_train)}")
    print(f"len(X_test): {len(X_test)}")

    # Multinomial Naive Bayes classifier inicialization
    pipeMNB = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])

    # Logistic regression inicialization
    pipeLR  = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(random_state=1979))])

    # Support Vector Classifier (SVC) inicialization    
    pipeSVC = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])

    param_grid_MNB = {'clf__alpha': np.linspace(0.1, 1.5, 10), 'clf__fit_prior': [True, False], 'clf__force_alpha': [True, False]}
    param_grid_LR = {'clf__C': np.logspace(-5, 8, 15)} 
    param_grid_SVC = {"clf__class_weight": [None, "balanced"], "clf__tol": np.logspace(-5, 8, 15)} 
    param_grid_tfidf = {
            "tfidf__binary": [True, False], 
            # "tfidf__norm": [None, 'l1', 'l2'], 
            "tfidf__lowercase": [True, False], 
            "tfidf__stop_words": [None, "english"],
            "tfidf__ngram_range": [(1,1), (2,2), (3,3)]}
    param_grid_MNB.update(param_grid_tfidf)
    param_grid_LR.update(param_grid_tfidf)
    param_grid_SVC.update(param_grid_tfidf)

    gridCV_MNB = GridSearchCV(pipeMNB, param_grid_MNB, cv=5, scoring="f1_macro")
    gridCV_LR = GridSearchCV(pipeLR, param_grid_LR, cv=5, scoring="f1_macro")
    gridCV_SVC = GridSearchCV(pipeSVC, param_grid_SVC, cv=5, scoring="f1_macro")

    print("\nAccuracy:", end="")
    mnb = gridCV_MNB.fit(X_train, y_train)
    predictMNB = gridCV_MNB.predict(X_test)
    print(f"\nMNB: {accuracy_score(y_test, predictMNB):.3f}")
    lr = gridCV_LR.fit(X_train, y_train)
    predictLR = gridCV_LR.predict(X_test)
    print(f"LR: {accuracy_score(y_test, predictLR):.3f}")
    gridCV_SVC.fit(X_train, y_train)
    svc = predictSVC = gridCV_SVC.predict(X_test)
    print(f"SVC: {accuracy_score(y_test, predictSVC):.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictSVC))

    print("\nMNB best params:")
    print("  Best Score: ", gridCV_MNB.best_score_)
    print("  Best Params: ", gridCV_MNB.best_params_)
    print("LR best params:")
    print("  Best Score: ", gridCV_LR.best_score_)
    print("  Best Params: ", gridCV_LR.best_params_)
    print("SVC best params:")
    print("  Best Score: ", gridCV_SVC.best_score_)
    print("  Best Params: ", gridCV_SVC.best_params_)

    print("\nCV Results:")
    print(pd.DataFrame(mnb.cv_results_))



if __name__ == "__main__":
    # classifiers()
    # classifiers_tuning()
    # pipeline_tuning()
    corpus_filename = "Dmoz-Science"

    df, X, y = read_dataset("./" + corpus_filename + ".csv")
    print("\nDataset content:")
    print(df.head(5))

    print("\nClasses couting:")
    print(df["class"].value_counts())

    print(f"\nDataset size: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=1979)
    print(f"len(X_train): {len(X_train)}")
    print(f"len(X_test): {len(X_test)}")

    pipeMNB = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])
    pipeLR  = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(random_state=1979, verbose=0))])
    pipeSVC = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])

    paramMNB = get_multinomial_naive_bayes_params("clf")
    paramLR = get_logistic_regression_params("clf")
    paramSVC = get_support_vector_params("clf")

    paramTfidf = get_tfidf_params("tfidf")
    # paramMNB.update(paramTfidf)
    # paramLR.update(paramTfidf)
    # paramSVC.update(paramTfidf)

# Multinomial Naive Bayes
    grid_search_MNB = fit_tuning(X_train, y_train, pipeMNB, paramMNB)

    print("\nMNB best params:")
    print("  Best Score: ", grid_search_MNB.best_score_)
    print("  Best Params: ", grid_search_MNB.best_params_)

    print("\nCV Results:")
    df_MNB = pd.DataFrame(grid_search_MNB.cv_results_)
    df_MNB = df_MNB.sort_values(by="mean_test_score", ascending=False)
    df_MNB

    # Save dataframe into CSV file
    df_MNB.to_csv(corpus_filename + "-mnb-results.csv")

# Logistic Regression
    grid_search_LR = fit_tuning(X_train, y_train, pipeLR, paramLR)

    print("\nLR best params:")
    print("  Best Score: ", grid_search_LR.best_score_)
    print("  Best Params: ", grid_search_LR.best_params_)

    print("\nCV Results:")
    df_LR = pd.DataFrame(grid_search_LR.cv_results_)
    df_LR = df_LR.sort_values(by="mean_test_score", ascending=False)
    df_LR
    
    # Save dataframe into CSV file
    df_LR.to_csv(corpus_filename + "-lr-results.csv")

# Support Vector Machine
    grid_search_SVC = fit_tuning(X_train, y_train, pipeSVC, paramSVC)

    print("\nSVC best params:")
    print("  Best Score: ", grid_search_SVC.best_score_)
    print("  Best Params: ", grid_search_SVC.best_params_)

    print("\nCV Results:")
    df_SVC = pd.DataFrame(grid_search_SVC.cv_results_)
    df_SVC = df_SVC.sort_values(by="mean_test_score", ascending=False)
    df_SVC

    # Save dataframe into CSV file
    df_SVC.to_csv(corpus_filename + "-svc-results.csv")
