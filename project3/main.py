# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import wandb
import random


PATH = 'dataset/'
BATCH_SIZE=64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TripletDataset(Dataset):
    def __init__(self, triplets, root_dir, file_to_embedding):
        self.triplets = triplets
        self.root_dir = root_dir
        self.file_to_embedding = file_to_embedding


    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx].strip().split(' ')
        
        anchor = self.file_to_embedding[anchor]
        positive = self.file_to_embedding[positive]
        negative = self.file_to_embedding[negative]

        return anchor, positive, negative


def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    print("generating embeddings...")
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("transforming dataset")
    train_dataset = datasets.ImageFolder(root=PATH, transform=train_transforms)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              pin_memory=True, num_workers=10)

    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    # get first 20 layers to create embedding
    model = nn.Sequential(*list(model.children())[0])
    for param in model.parameters():
        param.requires_grad = False
    embeddings = []
    batches = len(train_loader)
    print("embedding...")
    model = model.to(device)
    for i, (batch, _) in enumerate(train_loader):
        print("Batch", f'{i}/{batches:2}')
        batch = batch.to(device)
        embedding = model(batch)
        embeddings.append(embedding.cpu().detach().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    # Print the shape of the embedding
    print("embedding shape:",embeddings.shape) 

    np.save(PATH + 'embeddings.npy', embeddings)


def create_loader(file, batch_size=BATCH_SIZE, shuffle=True, val = False, num_workers = 4):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    print("getting data...")
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)
            
    embeddings = np.load(PATH + 'embeddings.npy')
    file_to_embedding = {}
    dataset = datasets.ImageFolder(root=PATH,
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in dataset.samples]
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]

    if val:
        data_len = len(dataset)
        val_size = int(data_len * 0.2)
        random.seed(42)
        random.shuffle(filenames)
        val_triplets = []
        val_count = 0
        for name in filenames:
            triplets_with_name = [triplet for triplet in triplets if name in triplet]
            val_triplets.append(triplets_with_name)
            val_count += len(triplets_with_name)
            triplets = [triplet for triplet in triplets if name not in triplet]
            if (val_count >= val_size):
                break
        
        val_triplets = np.hstack(val_triplets)
        train_triplets = triplets
        train_set = TripletDataset(triplets=train_triplets, root_dir = PATH, file_to_embedding=file_to_embedding)
        val_set = TripletDataset(triplets=val_triplets, root_dir = PATH, file_to_embedding=file_to_embedding)
        train_loader = DataLoader(dataset=train_set,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
        val_loader = DataLoader(dataset=val_set,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    else:
        dataset = TripletDataset(root_dir=PATH, triplets=triplets, file_to_embedding=file_to_embedding)
        train_loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
        val_loader = None
    return train_loader, val_loader
    
def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()

class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.last_conv_layer = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc1 = nn.Linear(25088, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        
    def _onefoward(self, x):
        x = self.last_conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.normalize(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.normalize(x)
        x = self.fc3(x)
        return x
        
    def forward(self, anchor, positive, negative):
        anchor = self._onefoward(anchor)
        positive = self._onefoward(positive)
        negative = self._onefoward(negative)
        
        return anchor, positive, negative

def train_model(train_loader, val_loader= None, val=False):
    model = Net()
    model.train()
    model = model.to(device)
    # log wandb
    wandb.watch(model, log="all")
    n_epochs = 30
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        model.train()
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()

            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        log = {"train loss": epoch_loss}
        val_loss = 0
        if val:
            model.eval()
            with torch.no_grad():
                false_predictions = []
                for anchor, pos, neg in val_loader:
                    anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                    anchor_emb, pos_emb, neg_emb = model(anchor, pos, neg)
                    distance_pos = (anchor_emb - pos_emb).pow(2).sum(1)
                    distance_neg = (anchor_emb - neg_emb).pow(2).sum(1)
                    f_pred = (distance_pos >= distance_neg).type(torch.int).cpu().numpy() # 1 if false, 0 if true
                    false_predictions.append(f_pred)
                false_predictions = np.hstack(false_predictions)
                false_pred_percent = false_predictions.sum(0) / len(false_predictions)
                log["val loss (wrong percent)"] = false_pred_percent
                val_loss = false_pred_percent

        print(f'epoch: {epoch:2}  loss: {epoch_loss:10.8f} validation loss: {val_loss:10.8f}')
        wandb.log(log)
        if epoch > 0 and epoch % 10 == 0:
            modelname = f'{epoch:2}model.pth'
            path = 'checkpoints/'+modelname
            torch.save(model.state_dict(), path)
            wandb.save(path)

    return model

def test_model(loader, model_exists):
    model = Net()
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)
    predictions = []
    model.eval()
    with torch.no_grad():
        for anchor, a, b in loader:
            anchor, a, b = anchor.to(device), a.to(device), b.to(device)
            anchor_emb, a_emb, b_emb = model(anchor, a, b)
            distance_a = (anchor_emb - a_emb).pow(2).sum(1)
            distance_b = (anchor_emb - b_emb).pow(2).sum(1)
            pred = (distance_a < distance_b).type(torch.int).cpu().numpy()
            predictions.append(pred)
        predictions = np.hstack(predictions)
    # save predicitons
    np.savetxt("results.txt", predictions, fmt='%i')
    if model_exists == False:
        wandb.save("results.txt")

    
def main():
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists(PATH + 'embeddings.npy') == False):
        generate_embeddings()

    # define a model and train it
    model_exists = os.path.exists('model.pth')
    if(model_exists== False):
        # init wandb
        wandb.init(project="project3IML")
        
        # Create train_data loader
        val = False
        train_loader, val_loader = create_loader(TRAIN_TRIPLETS, batch_size=BATCH_SIZE, shuffle= True, val = val)
        model = train_model(train_loader, val_loader, val=val)
    
        # save model
        torch.save(model.state_dict(), 'model.pth')
        wandb.save('model.pth')
    
    # test the model on the test data
    test_loader, _ = create_loader(TEST_TRIPLETS, batch_size=BATCH_SIZE, shuffle=False)
    test_model(test_loader, model_exists)

# Main function. You don't have to change this
if __name__ == '__main__':
    main()