import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV


def create_prefix(prefix):
    if prefix:
        return prefix + "__"
    else:
        return ''


def get_logistic_regression_params(prefix=None):
    """
    Generate hyperparameter options for tuning a Logistic Regression model.

    This function creates a dictionary of hyperparameters that can be used 
    for fine-tuning a Logistic Regression model during grid search or other
    hyperparameter optimization methods. The parameter names can be prefixed 
    to allow compatibility with pipelines or other naming conventions.

    Args:
        prefix (str, optional): A string to be prefixed to the parameter names. 
                                Defaults to None.

    Returns:
        dict: A dictionary containing hyperparameter options for Logistic Regression.
              The keys are parameter names (optionally prefixed), and the values are 
              the corresponding ranges or options.
              
              - 'C': Regularization strength, with values on a logarithmic scale 
                between 10^-5 and 10^8.
              - 'solver': Solvers used in the optimization process, such as 'liblinear' 
                and 'lbfgs'.
              - 'class_weight': Specifies class balancing options, either None or 'balanced'.
              - 'max_iter': Maximum number of iterations for the solver, with values 
                100, 200, and 500.

    Example:
        >>> get_logistic_regression_params(prefix="lr__")
        {
            'lr__C': array([1.00000000e-05, ..., 1.00000000e+08]),
            'lr__solver': ['liblinear', 'lbfgs'],
            'lr__class_weight': [None, 'balanced'],
            'lr__max_iter': [100, 200, 500]
        }
    """    
    prefix = create_prefix(prefix)
    param = {
        prefix + 'C': np.logspace(-5, 8, 10),  # Refine range of regularization
        prefix + 'solver': ['liblinear', 'lbfgs'],
        prefix + 'class_weight': [None, 'balanced'],
        prefix + 'max_iter': [100, 200, 500]
        # prefix + 'penalty': ['l2'],
        # prefix + 'tol': np.logspace(-4, -1, 4)
    }    
    return param
    
    
def get_support_vector_params(prefix=None):
    """
    Generate hyperparameters for tuning a support vector classifier.
    
    Parameters:
    - prefix (str, optional): A string to prepend to the parameter names for compatibility with pipelines. 
                              Defaults to None.
                              
    Returns:
    - dict: A dictionary containing the hyperparameter grid for SVC.
            Includes:
            - 'class_weight': Options for None or "balanced".
            - 'tol': Tolerance levels sampled logarithmically over 15 values between 1e-5 and 1e8.
    """
    prefix = create_prefix(prefix)
    param  = {
            prefix + "class_weight": [None, "balanced"], 
            prefix + "tol": np.logspace(-5, 8, 15)
        } 
    return param


def get_multinomial_naive_bayes_params(prefix=None):
    """
    Generate hyperparameters for tuning a multinomial Naive Bayes model.
    
    Parameters:
    - prefix (str, optional): A string to prepend to the parameter names for compatibility with pipelines. 
                              Defaults to None.
                              
    Returns:
    - dict: A dictionary containing the hyperparameter grid for multinomial Naive Bayes.
            Includes:
            - 'alpha': Range of smoothing parameters between 0.1 and 1.5 (10 evenly spaced values).
            - 'fit_prior': Options for True or False.
            - 'force_alpha': Options for True or False.
    """
    prefix = create_prefix(prefix)
    param = {
            prefix + 'alpha': np.linspace(0.1, 1.5, 10), 
            prefix + 'fit_prior': [True, False], 
            prefix + 'force_alpha': [True, False]
        }
    return param


def get_tfidf_params(prefix=None):
    """
    Generate hyperparameters for tuning a TF-IDF vectorizer.
    
    Parameters:
    - prefix (str, optional): A string to prepend to the parameter names for compatibility with pipelines. 
                              Defaults to None.
                              
    Returns:
    - dict: A dictionary containing the hyperparameter grid for TF-IDF vectorization.
            Includes:
            - 'binary': Options for binary term frequency (True or False).
            - 'lowercase': Options to convert text to lowercase (True or False).
            - 'stop_words': Options for None or "english".
            - 'ngram_range': Range of n-grams to extract (e.g., unigrams, bigrams, trigrams).
    """
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
    """
    Perform hyperparameter tuning using GridSearchCV with 3-fold cross-validation.
    
    Parameters:
    - X_train (pd.DataFrame or np.ndarray): Training feature set.
    - y_train (pd.Series or np.ndarray): Training labels.
    - estimator (sklearn estimator): Machine learning model to be tuned.
    - param_grid (dict): Dictionary of hyperparameters to test.
    
    Returns:
    - sklearn.model_selection.GridSearchCV: Fitted GridSearchCV object with the best parameters.
    """
    grid_search = GridSearchCV(estimator, param_grid, cv=3, scoring="f1_macro")
    grid_search.fit(X_train, y_train)
    return grid_search


def read_dataset(dataset_filename_csv):
    """
    Load a dataset from a CSV file and extract features and labels.
    
    Parameters:
    - dataset_filename_csv (str): Path to the CSV file containing the dataset.
    
    Returns:
    - tuple: (pd.DataFrame, pd.Series, pd.Series)
             - Full dataset as a DataFrame.
             - Text features (X) as a Series.
             - Labels (y) as a Series.
    """
    dataset = pd.read_csv(dataset_filename_csv)
    X = dataset["text"]
    y = dataset["class"]
    return dataset, X, y


def show_score_params(grid_search):
    """
    Display the best parameters and score from a GridSearchCV object.

    Parameters
    ----------
    grid_search : GridSearchCV
        The fitted GridSearchCV object containing the results of the hyperparameter search.

    Returns
    -------
    None
        Prints the best score and parameters to the console.
    """
    print("\nSVC best params:")
    print(f"  Best Score: {grid_search.best_score_}")
    print(f"  Best Params: {grid_search.best_params_}")


def results_to_csv(grid_search, filename, verbose=False):
    """
    Save the cross-validation results of a GridSearchCV object to a CSV file.

    Parameters
    ----------
    grid_search : GridSearchCV
        The fitted GridSearchCV object containing the cross-validation results.
    filename : str
        The path to the CSV file where the results should be saved.
    verbose : bool, optional
        If True, prints the cross-validation results to the console before saving. Default is False.

    Returns
    -------
    None
        Saves the sorted cross-validation results to a CSV file.
    """    
    print("\nCV Results:", verbose)
    df_SVC = pd.DataFrame(grid_search.cv_results_)
    df_SVC = df_SVC.sort_values(by="mean_test_score", ascending=False)
    print(df_SVC, verbose)
    # Save dataframe into CSV file
    df_SVC.to_csv(filename)


def validate(estimator, X_test, y_test):
    """
    Validate a trained estimator using the test dataset.

    Parameters:
    estimator (object): The trained model implementing a `predict` method.
    X_test (array-like): Test feature dataset.
    y_test (array-like): True labels for the test dataset.

    Returns:
    list: A list containing the following metrics:
        - Accuracy score
        - F1-score (macro-averaged)
        - F1-score (micro-averaged)
    """
    predicted = estimator.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    f1_macro = f1_score(y_test, predicted, average='macro')
    f1_micro = f1_score(y_test, predicted, average='micro') 
    metrics = [accuracy, f1_macro, f1_micro]
    return metrics


if __name__ == "__main__":
    print("SVC parameters:")
    print(get_support_vector_params())

    print("TF-IDF parameters:")
    print(get_tfidf_params("tfidf"))   

    print(np.linspace(100, 1000, 3)) 
