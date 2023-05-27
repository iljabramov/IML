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
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

FEATURE_BATCH_SIZE = 10000
FEATURE_EPOCHS = 500
FEATURE_LR = 0.002

SMALL_BATCH_SIZE = 1
SMALL_EPOCHS = 200
SMALL_LR = 0.002

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
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 32)
        self.fc9 = nn.Linear(32, 16)
        self.fc10 = nn.Linear(16, 1)

    def forward(self, x):
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
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc8(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc9(x)
        x = F.leaky_relu(x)
        x = self.fc10(x)
        return x
    
def feature_extractor_model(val= True):
    """
    This function trains the feature extractor on the pretraining data.

    input:  batch_size: int, the batch size used for training
            eval_size: int, the size of the validation set
            
    """
    x = pd.read_csv("dataset/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y = pd.read_csv("dataset/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    print("Pretrain data loaded!")
    
    model = Feature_Net().to(device)
    train(x,y, model, "feature",FEATURE_EPOCHS, FEATURE_BATCH_SIZE, FEATURE_LR, val)

def test():
    x = pd.read_csv("dataset/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    print(f"Testing model...")
    
    

    model = Feature_Net().to(device)
    model.load_state_dict(torch.load('small.pth', map_location=device))

    model.eval()
    with torch.no_grad():
        y_pred = torch.flatten(model(torch.tensor(x.to_numpy(), dtype=torch.float)))
    

    assert y_pred.shape == (x.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")
    return

def train(x, y, model, name, epochs, batchsize, lr, val):
    if val:
        x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
        x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
        y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
        train_dataset = TensorDataset(x_tr,y_tr)
        val_dataset = TensorDataset(x_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
    else: 
        x, y = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
        train_dataset = TensorDataset(x,y)

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    print(f"Training {name} model...")

        
    wandb.watch(model, log="all")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_train_loss = 0.
        train_batches = 0
        
        model.train()
        for data in train_dataloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred = torch.flatten(y_pred)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            train_batches += 1
        av_train_loss = running_train_loss / train_batches
        wandblog = {name + "_training_loss": av_train_loss}
        
        if val:
            running_val_loss = 0.
            val_batches = 0
            model.eval()
            with torch.no_grad():
                for data in val_dataloader:
                    x, y = data
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    y_pred = torch.flatten(y_pred)
                    loss = loss_fn(y_pred, y)
    
                    running_val_loss += loss.item()
                    val_batches += 1
            av_val_loss = running_val_loss / val_batches
            wandblog[name + "_validation_loss"] = av_val_loss
        
        wandb.log(wandblog)

        if epoch % 10 == 9:
            print(f"epoch {epoch + 1} done.")
            
    path = name + ".pth"
    torch.save(model.state_dict(), path)
    wandb.save(path)
    print("Done.")
    return

def train_model(val = True):
    # load data and feature model
    x = pd.read_csv("dataset/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y = pd.read_csv("dataset/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    
    model = Feature_Net().to(device)
    model.load_state_dict(torch.load('feature.pth', map_location=device))
    #for name, param in model.named_parameters():
    #    if name in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias', 'fc4.weight', 'fc4.bias']:
    #        param.requires_grad = False

    
    train(x, y, model, "small", SMALL_EPOCHS, SMALL_BATCH_SIZE, SMALL_LR, val)



def main():
    #delete_models()
    
    if not os.path.exists('feature.pth') or not os.path.exists('small.pth'):
        wandb.init(project="project4IML")
    
    if not os.path.exists('feature.pth'):
        feature_extractor_model(val = True)
    
    if not os.path.exists('small.pth'):
        train_model(val = True)

    if not os.path.exists('results.csv'):
        test()
    
if __name__ == '__main__':
    main()