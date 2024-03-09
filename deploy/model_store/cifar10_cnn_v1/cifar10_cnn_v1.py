import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

IMG_SIZE = 32

CLASS_LABELS = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Creating dictionery for associating labels from output
OP_TO_LABELS = {i: j for i, j in zip(range(len(CLASS_LABELS)), CLASS_LABELS)}

MODEL_PATH = './model_store/cifar10_cnn_v1/cifar10_cnn_v1.pth'


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def load_model(device):
    
    print("INFERENCE DEVICE :", device)

    model = CNNModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    return model

# Input Image and preprocess
def filenameToPILImage(x): return Image.open(x).convert('RGB')

infer_transform = transforms.Compose([filenameToPILImage, transforms.Resize(
    (IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])