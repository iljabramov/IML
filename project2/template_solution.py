# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import math
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # drop seasons because they have little correlation with price
    tr_num = train_df.drop("season", axis = 1)
    test_num = test_df.drop("season", axis = 1)

    # add missing features
    imputer = KNNImputer(n_neighbors=5)
    tr_num = pd.DataFrame(imputer.fit_transform(tr_num), columns= tr_num.keys())
    test_num = pd.DataFrame(imputer.fit_transform(test_num), columns= test_num.keys())

    # get lables and data
    y_train = tr_num["price_CHF"].to_frame()
    X_train = tr_num.drop("price_CHF", axis = 1)
    X_test = test_num

    # transform data
    std_scaler = StandardScaler()
    X_train = pd.DataFrame(std_scaler.fit_transform(X_train), columns= X_train.keys())
    X_test = pd.DataFrame(std_scaler.fit_transform(X_test), columns= X_test.keys())

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def train_model(model, X, y, epochs = 40, lr = 0.001):
  # define our loss and optimizer
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # train the model
  model.train()

  for epoch in range(epochs):
      running_loss = 0.0
      for i, data in enumerate(X):
          optimizer.zero_grad()

          data = data.float().to(device)
          pred = model(data)
          loss = criterion(pred, y[i].to(torch.float).to(device))
          loss.backward()
  
          optimizer.step()

          running_loss += loss.item()
      print(f'Epoch: {epoch + 1} loss: {running_loss / len(X):.5f}')

      running_loss = 0.0

def modeling_and_prediction(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    model = torch.nn.Sequential(
        torch.nn.Linear(9, 24),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(24, 9),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(9, 1),
        torch.nn.LeakyReLU(),
    )

    train_model(model, torch.Tensor(X_train.to_numpy()), torch.Tensor(y_train.to_numpy()), 200)

    y_pred = model(torch.Tensor(X_test.to_numpy())).flatten()

    assert y_pred.shape == (100,), y_pred.shape
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # The function retrieving optimal LR parameters
    y_pred = modeling_and_prediction(X_train, y_train, X_test)
    # score = r2_score(y_train_val, y_pred.detach().numpy())
    # print(score)
    # Save results in the required format
    dt = pd.DataFrame(y_pred.detach().numpy()) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
