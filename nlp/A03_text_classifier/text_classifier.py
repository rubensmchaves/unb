import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from find_best_hyperparameters import fit_tuning
from find_best_hyperparameters import get_multinomial_naive_bayes_params
from find_best_hyperparameters import get_support_vector_params
from find_best_hyperparameters import get_logistic_regression_params
from find_best_hyperparameters import get_tfidf_params
from find_best_hyperparameters import read_dataset
from find_best_hyperparameters import show_score_params
from find_best_hyperparameters import results_to_csv
from find_best_hyperparameters import validate


if __name__ == "__main__":
    corpus_filename = "Dmoz-Science"

    df, X, y = read_dataset(f"./data/{corpus_filename}.csv")
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
    show_score_params(grid_search_MNB)
    results_to_csv(grid_search_MNB, f"./results/{corpus_filename}-mnb-results.csv")
    estimator = grid_search_MNB.best_estimator_
    print(estimator)
    estimator.fit(X_train, y_train)
    mnb_metrics = validate(estimator, X_test, y_test)

# Logistic Regression
    grid_search_LR = fit_tuning(X_train, y_train, pipeLR, paramLR)
    show_score_params(grid_search_LR)
    results_to_csv(grid_search_LR,  f"./results/{corpus_filename}-lr-results.csv")
    estimator = grid_search_LR.best_estimator_
    print(estimator)
    estimator.fit(X_train, y_train)
    lr_metrics = validate(estimator, X_test, y_test)

# Support Vector Machine
    grid_search_SVC = fit_tuning(X_train, y_train, pipeSVC, paramSVC)
    show_score_params(grid_search_SVC)
    results_to_csv(grid_search_SVC,  f"./results/{corpus_filename}-svc-results.csv")
    estimator = grid_search_SVC.best_estimator_
    print(estimator)
    estimator.fit(X_train, y_train)
    svc_metrics = validate(estimator, X_test, y_test)

# Print all resulted metrics from validation
    results = { 
        "Metrics": ["Accuracy", "F1 macro", "F1 micro"],        
        "MNB": mnb_metrics,
        "LR": lr_metrics,
        "SVC": svc_metrics
    }
    print(pd.DataFrame(results))
