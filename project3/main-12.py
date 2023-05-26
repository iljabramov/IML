import numpy as np
import os
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import wandb
import random
from PIL import Image


BATCH_SIZE = 59515
EPOCHS = 200
LR = 0.002

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


def delete_models():
    for file in os.listdir("."):
        if file.startswith("model_"):
            os.remove(file)


class TrainTriplets(Dataset):
    def __init__(self):
        super().__init__()
        self.embeds = np.load("embeddings.npy")
        with open("train_triplets.txt") as f:
            self.instances = [tuple(map(int, line.split())) for line in f.readlines()]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        anchor, pos, neg = self.instances[idx]
        x_anchor = self.embeds[anchor]
        x_pos = self.embeds[pos]
        x_neg = self.embeds[neg]

        if random.randint(0, 1) == 1:
            x = torch.from_numpy(np.concatenate((x_anchor, x_pos, x_neg))).requires_grad_().to(device)
            y = torch.tensor(1, device=device, dtype=torch.float)
        else:
            x = torch.from_numpy(np.concatenate((x_anchor, x_neg, x_pos))).requires_grad_().to(device)
            y = torch.tensor(0, device=device, dtype=torch.float)
        
        return x, y


class TestTriplets(Dataset):
    def __init__(self):
        super().__init__()
        self.embeds = np.load("embeddings.npy")
        with open("test_triplets.txt") as f:
            self.instances = [tuple(map(int, line.split())) for line in f.readlines()]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        a, b, c = self.instances[idx]
        x_a = self.embeds[a]
        x_b = self.embeds[b]
        x_c = self.embeds[c]
        
        return torch.from_numpy(np.concatenate((x_a, x_b, x_c))).to(device).to(torch.float32)


class JonasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6144, 6144)
        self.fc2 = nn.Linear(6144, 3072)
        self.fc3 = nn.Linear(3072, 1536)
        self.fc4 = nn.Linear(1536, 1536)
        self.fc5 = nn.Linear(1536, 1536)
        self.fc6 = nn.Linear(1536, 1536)
        self.fc7 = nn.Linear(1536, 1536)
        self.fc8 = nn.Linear(1536, 1536)
        self.fc9 = nn.Linear(1536, 1536)
        self.fc10 = nn.Linear(1536, 1536)
        self.fc11 = nn.Linear(1536, 1536)
        self.fc12 = nn.Linear(1536, 1536)
        self.fc13 = nn.Linear(1536, 1536)
        self.fc14 = nn.Linear(1536, 512)
        self.fc15 = nn.Linear(512, 124)
        self.fc16 = nn.Linear(124, 36)
        self.fc17 = nn.Linear(36, 12)
        self.fc18 = nn.Linear(12, 1)

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
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc10(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc11(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc12(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc13(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc14(x)        
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc15(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc16(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc17(x)
        x = F.dropout(x, p=0.2)
        x = F.leaky_relu(x)
        x = self.fc18(x)
        x = F.sigmoid(x)
        return x


def test():
    print("Generating results...")
    test_dataloader = DataLoader(TestTriplets(), batch_size=59544, shuffle=False)
    model = JonasModel().to(device)
    model.load_state_dict(torch.load('model_best.pth', map_location=device))
    model.eval()

    preds = list()
    for x in test_dataloader:
        pred_y = model(x)
        pred_y = torch.flatten(pred_y).tolist()
        preds.extend(map(lambda y: 0 if y < 0.5 else 1, pred_y))

    np.savetxt("results.txt", preds, fmt='%i')
    wandb.save("results.txt")
    print("Done.")

def train():
    print("Training model...")
    train_dataloader = DataLoader(TrainTriplets(), batch_size=BATCH_SIZE, shuffle=True)

    model = JonasModel().to(device)
    wandb.watch(model, log="all")
    model.train()

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=LR)

    best_loss = float("inf")
    for epoch in range(EPOCHS):
        running_loss = 0.
        for data in train_dataloader:
            triplet, y = data

            optimizer.zero_grad()

            output = model(triplet)
            output = torch.flatten(output)

            loss = loss_fn(output, y)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        wandb.log({"training_loss": running_loss})

        if 20 <= epoch and (running_loss < best_loss or epoch % 10 == 9):
            path = f"model_{epoch}.pth"
            best_loss = min(best_loss, running_loss)
            torch.save(model.state_dict(), path)
            wandb.save(path)
            os.system(f"cp {path} model_best.pth")

        if epoch % 10 == 9:
            print(f"epoch {epoch + 1} done.")

    wandb.save("model_best.pth")
    print("Done.")

def make_embeds():
    print("Generating embeddings...")
    dataset = os.listdir("dataset/food")
    to_tens = transforms.ToTensor()
    model = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).children())[:-1]).to(device)
    model.eval()
    embeds = [None for _ in dataset]
    with torch.no_grad():
        for i, img_name in enumerate(dataset):
            if i%1000 == 999:
                print(i+1)
            pil = Image.open(os.path.join("dataset/food", img_name))
            tens = to_tens(pil)
            tens = tens.to(device).reshape((1, *tens.shape))
            embed = model(tens).reshape((2048,))
            embeds[int(img_name.replace(".jpg", ""))] = embed.cpu().numpy()
    embeds = np.array(embeds)
    np.save('embeddings.npy', embeds)
    print("Done.")

def main():
    # delete_models()

    if not os.path.exists('embeddings.npy'):
        make_embeds()

    wandb.init(project="project3IML")

    if not os.path.exists('model_best.pth'):
        train()

    test()


if __name__ == '__main__':
    main()
