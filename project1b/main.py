# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
def fit(X: np.ndarray, y: np.ndarray, lam: int):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned. 

    Parameters
    ----------
    X: matrix of floats, dim = (135,13), inputs with 13 features
    y: array of floats, dim = (135,), input labels)
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """
    I = np.eye(X.shape[1])
    w = np.linalg.inv(X.T@X + lam*I) @ (X.T@y)
    assert w.shape == (21,)
    return w


def calculate_RMSE(w: np.ndarray, X: np.ndarray, y: np.ndarray):
    """This function takes test data points (X and y), and computes the empirical RMSE of 
    predicting y from X using a linear model with weights w. 

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression 
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """
    RMSE = np.sqrt(1/len(y) * (np.linalg.norm(X.dot(w) - y)**2 ))
    assert np.isscalar(RMSE)
    return RMSE


def average_LR_RMSE(X, y, lambdas, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated, 
    and then averaged over iterations.
    
    Parameters
    ---------- 
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV
    
    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """
    avg_RMSE = np.zeros(5)
    kf = KFold(n_splits=n_folds, shuffle=False)
    kf.get_n_splits(X)
    for i in range(len(lambdas)):
        sum_RMSE = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            w = fit(X_train, y_train, lambdas[i])
            sum_RMSE += calculate_RMSE(w, X_test, y_test)
        avg_RMSE[i] = sum_RMSE / n_folds
        
    assert avg_RMSE.shape == (5,)
    return avg_RMSE

def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    
    for i in range(len(X_transformed)):
        X_transformed[i][:5] += X[i]
        X_transformed[i][5:10] += X[i]*X[i]
        X_transformed[i][10:15] += np.exp(X[i])
        X_transformed[i][15:20] += np.cos(X[i])
        X_transformed[i][20] = 1
        
    assert X_transformed.shape == (700, 21)
    return X_transformed


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    X = data.to_numpy()
    
    lambdas = [0, 0.1, 1, 10, 100]
    n_folds = 10
    X_transformed = transform_data(X)
    avg_RMSE = average_LR_RMSE(X_transformed, y, lambdas, n_folds)
    print(avg_RMSE)
    
    
    # The function retrieving optimal LR parameters
    w = fit(X_transformed, y, lambdas[np.argmin(avg_RMSE)])
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
