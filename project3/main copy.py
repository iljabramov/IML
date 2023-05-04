import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import VGG16_Weights
from PIL import Image

PATH = 'dataset/food/'
BATCH_SIZE = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data(file, train=True):
    print("getting data...")
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line.strip().split())

    return triplets

class TripletDataset(Dataset):
    def __init__(self, triplets, root_dir, transform=None):
        self.triplets = triplets
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]
        anchor = self._load_image(os.path.join(self.root_dir, anchor_path + '.jpg'))
        positive = self._load_image(os.path.join(self.root_dir, positive_path + '.jpg'))
        negative = self._load_image(os.path.join(self.root_dir, negative_path + '.jpg'))

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def _load_image(self, path):
        return Image.open(path).convert('RGB')

def create_loader_from_np(triplets, root_dir, train=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=4):
    print("creating loader...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = TripletDataset(triplets, root_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

class TripletRankingModel(nn.Module):
    def __init__(self):
        super(TripletRankingModel, self).__init__()
        self.base_network = self.get_base_network()

    def get_base_network(self):
        model = models.vgg16(weights = VGG16_Weights.DEFAULT)
        model = nn.Sequential(*list(model.children())[:-1])
        return model

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.base_network(anchor)
        positive_embedding = self.base_network(positive)
        negative_embedding = self.base_network(negative)
        return anchor_embedding, positive_embedding, negative_embedding

def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()

def train_model(train_loader):
    model = TripletRankingModel()
    model.train()
    model.to(device)
    n_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()

            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f'epoch: {epoch:2}  loss: {epoch_loss:10.8f}')

    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # load the training and testing data
    train_triplets = get_data(TRAIN_TRIPLETS)
    test_triplets = get_data(TEST_TRIPLETS, train=False)

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(train_triplets, PATH, train=True, batch_size=BATCH_SIZE)
    test_loader = create_loader_from_np(test_triplets, PATH, train=False, batch_size=2048, shuffle=False)

    # define a model and train it
    model = train_model(train_loader)

    # TODO: Implement the test_model function to test the model on the test data
    # test_model(model, test_loader)
    # print("Results saved to results.txt")

