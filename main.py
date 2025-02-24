import torch
import torch.optim as optim
from dataset import MiniImagenet
from maml import MAML
from model import SimpleCNN
from torch.utils.data import Dataset, DataLoader
from trainer import train_maml, evaluate_maml

root = "D:/Downloads/mini-imagenet"
dataset = MiniImagenet(root, mode='train', batchsz=32, n_way=5, k_shot=1, k_query=15, resize=84)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
dataset = MiniImagenet(root, mode='test', batchsz=32, n_way=5, k_shot=1, k_query=15, resize=84)
test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
base_model = SimpleCNN(n_way=5)
checkpoint = torch.load("maml_miniimagenet.pth")
base_model.load_state_dict(checkpoint)
maml = MAML(base_model, inner_lr=0.01, inner_steps=5)
meta_optimizer = optim.Adam(maml.model.parameters(), lr=0.001)
# Training
# train_maml(maml, train_loader, test_loader, meta_optimizer, epochs=200)
# Evaluate 

evaluate_maml(maml, test_loader)