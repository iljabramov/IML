# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

BATCH_SIZE = 59515
EPOCHS = 200
LR = 0.002

def delete_models():
    for file in os.listdir("."):
        if file.endswith(".pth"):
            os.remove(file)

class Feature_Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 124)
        self.fc6 = nn.Linear(124, 36)
        self.fc7 = nn.Linear(36, 12)
        self.fc8 = nn.Linear(12, 1)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.fc1(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc5(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc6(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x
    
def feature_extractor_model(batch_size=256, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data.

    input:  batch_size: int, the batch size used for training
            eval_size: int, the size of the validation set
            
    """
    x = pd.read_csv("dataset/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y = pd.read_csv("dataset/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    print("Pretrain data loaded!")
    
    
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=42, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
    train_dataset = TensorDataset(x_tr,y_tr)
    val_dataset = TensorDataset(x_val, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    
    print("Training Feature model...")
    model = Feature_Net().to(device)
    wandb.watch(model, log="all")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=LR)

    best_loss = float("inf")
    for epoch in range(EPOCHS):
        running_train_loss = 0.
        running_val_loss = 0.
        train_batches = 0
        val_batches = 0
        
        model.train()
        for data in train_dataloader:
            x, y = data
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            train_batches += 1
        
        model.eval()
        with torch.no_grad:
            for data in val_dataloader:
                x, y = data
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
    
                running_val_loss += loss.item()
                val_batches += 1
        
        av_train_loss = running_train_loss / train_batches
        av_val_loss = running_val_loss / val_batches
        wandb.log({"training_loss": av_train_loss, "validation_loss": av_val_loss})

        if 20 <= epoch and (av_train_loss < best_loss or epoch % 10 == 9):
            path = f"model_{epoch}.pth"
            best_loss = min(best_loss, av_train_loss)
            torch.save(model.state_dict(), path)
            wandb.save(path)
            os.system(f"cp {path} feature_extractor.pth")

        if epoch % 10 == 9:
            print(f"epoch {epoch + 1} done.")

    wandb.save("feature_extractor.pth")
    print("Done.")



    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        return x

    return make_features

def test():
    x_test = pd.read_csv("dataset/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")
    return

def train_model():
    # load data
    x_train = pd.read_csv("dataset/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("dataset/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    # regression model
    regression_model = get_regression_model()

def main():
    #delete_models()
    
    if not os.path.exists('feature_extractor.pth'):
        wandb.init(project="project4IML")
        feature_extractor_model()
    
    #if not os.path.exists('model.pth'):
        #train_model()

    #test()
    
if __name__ == '__main__':
    main()