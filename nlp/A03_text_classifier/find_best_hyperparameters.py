import numpy as np

from sklearn.model_selection import GridSearchCV


def create_prefix(prefix):
	if prefix:
		return prefix + "__"
	else:
		return ''


def get_logistic_regression_params(prefix=None):
	prefix = create_prefix(prefix)
	param = {prefix + 'C': np.logspace(-5, 8, 15)} 
	return param


def get_support_vector_params(prefix=None):
	prefix = create_prefix(prefix)
	param  = {
			prefix + "class_weight": [None, "balanced"], 
			prefix + "tol": np.logspace(-5, 8, 15)
		} 
	return param


def get_multinomial_naive_bayes_params(prefix=None):
	prefix = create_prefix(prefix)
	param = {
			prefix + 'alpha': np.linspace(0.1, 1.5, 10), 
			prefix + 'fit_prior': [True, False], 
			prefix + 'force_alpha': [True, False]
		}
	return param


def get_tfidf_params(prefix=None):
	prefix = create_prefix(prefix)
	param = {
			prefix + "binary": [True, False], 
			# prefix + "tfidf__norm": [None, 'l1', 'l2'], 
			prefix + "lowercase": [True, False], 
			prefix + "stop_words": [None, "english"],
			prefix + "ngram_range": [(1,1), (2,2), (3,3)]
		}
	return param


def fit_tuning(X_train, y_train, estimator, param_grid):
	grid_search = GridSearchCV(estimator, param_grid, cv=3, scoring="f1_macro")
	grid_search.fit(X_train, y_train)
	return grid_search



if __name__ == "__main__":
	print("SVC parameters:")
	print(get_support_vector_params())


	print("TF-IDF parameters:")
	print(get_tfidf_params("tfidf"))	