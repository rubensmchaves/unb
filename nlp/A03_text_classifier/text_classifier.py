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
from find_best_hyperparameters import get_tfidf_params
from find_best_hyperparameters import read_dataset


# Accuracy:
# MNB: 0.611
# LR: 0.711
# SVC: 0.844
def classifiers():
	df, X, y = read_dataset("./CSTR.csv")

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
	df, X, y = read_dataset("corpus/CSTR.csv")

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
	df, X, y = read_dataset("corpus/CSTR.csv")

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
	classifiers()
	# classifiers_tuning()
	# pipeline_tuning()

	# df, X, y = read_dataset("corpus/CSTR.csv")

	# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=1979)

	# pipeMNB = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])
	# pipeLR  = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(random_state=1979))])
	# pipeSVC = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])

	# param = get_multinomial_naive_bayes_params("clf")
	# param.update(get_tfidf_params("tfidf"))
	# grid_search = fit_tuning(X_train, y_train, pipeMNB, param)

	# print("\nMNB best params:")
	# print("  Best Score: ", grid_search.best_score_)
	# print("  Best Params: ", grid_search.best_params_)

	# print("\nCV Results:")
	# print(pd.DataFrame(grid_search.cv_results_))
