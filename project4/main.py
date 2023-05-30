# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import wandb
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.metrics import mean_squared_error


from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

FEATURE_EPOCHS = 80
FEATURE_LR = 0.01
FEATURE_BOOST_ROUND = 80

SMALL_EPOCHS = 30
SMALL_LR = 0.05
SMALL_BOOST_ROUND = 5

#FREEZE = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']

def delete_models():
    for file in os.listdir("."):
        if file.endswith(".model"):
            os.remove(file)

def feature_extractor_model(val= True, val_size = 0.2):
    """
    This function trains the feature extractor on the pretraining data.

    input:  batch_size: int, the batch size used for training
            eval_size: int, the size of the validation set
            
    """
    x = pd.read_csv("dataset/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y = pd.read_csv("dataset/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    print("Pretrain data loaded!")
    
    train(x,y, None, "feature",FEATURE_EPOCHS, FEATURE_LR, val, val_size, FEATURE_BOOST_ROUND)

def test():
    x_frame = pd.read_csv("dataset/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    x = x_frame.to_numpy()
    print(f"Testing model...")
    
        
    # Initialize a new booster
    booster = xgb.Booster()

    # Load the model from the file
    booster.load_model('small.model')
    y_pred = booster.predict(xgb.DMatrix(x))
    

    assert y_pred.shape == (x.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_frame.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")
    return

def train(x, y, path, name, epochs, lr, val, val_size, num_boost_round):
    wandb.init(project="project4IML", group="tweak-params-lyric-dawn-113-fresh-haze-112")
    
    # Define and log parameters
    config = {
        'FEATURE_EPOCHS' : FEATURE_EPOCHS,
        'FEATURE_LR' : FEATURE_LR,

        'SMALL_EPOCHS' : SMALL_EPOCHS,
        'SMALL_LR': SMALL_LR,
    }
    wandb.config.update(config)
    wandb.save("main.py")

    if val:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size, random_state=22, shuffle=True)
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
    else:
        x_train, y_train = x, y
        dtrain = xgb.DMatrix(x, label=y)

    params = {
        'n_jobs': 1,
        'objective':'reg:squarederror',
        'colsample_bytree': 0.7,
        'learning_rate': lr,
        'max_depth': 9,
        'alpha': 0.01
    }

    def evals_results():
        train_preds = model.predict(dtrain)
        train_loss = ((train_preds - y_train) ** 2).mean()

        if val:
            val_preds = model.predict(dval)
            val_loss = ((val_preds - y_val) ** 2).mean()
            return train_loss, val_loss
        else:
            return train_loss, None
    
    if path is not None:
        model = xgb.Booster()
        # Load the model from the file
        model.load_model('feature.model')
    else:
        model = xgb.train(params, dtrain, num_boost_round=1)
    
    # Train the model
    for epoch in range(epochs):
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round, xgb_model=model)
        
        train_loss, val_loss = evals_results()

        wandb.log({
            f"{name}_train_loss": train_loss,
            f"{name}_val_loss": val_loss
        })
        print(epoch)

    # Save the model
    model.save_model(name + ".model")

    print("Done.")
    wandb.finish()
    return



def train_model(val = True, val_size = 0.2):
    # load data and feature model
    x = pd.read_csv("dataset/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y = pd.read_csv("dataset/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    
    train(x, y, "feature.model", "small", SMALL_EPOCHS, SMALL_LR, val, val_size, SMALL_BOOST_ROUND)



def main():
    #delete_models()
  
    if not os.path.exists('feature.model'):
        feature_extractor_model(val = False, val_size= 0.1)
    
    if not os.path.exists('small.model'):
        train_model(val = False, val_size= 0.2)

    if not os.path.exists('results.csv'):
        test()
    
if __name__ == '__main__':
    main()