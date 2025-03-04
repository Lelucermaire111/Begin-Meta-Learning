import torch
import torch.optim as optim
from dataset import MiniImagenet
from maml import MAML
from model import SimpleCNN
from torch.utils.data import Dataset, DataLoader
from trainer import train_maml, evaluate_maml
from tensorboardX import SummaryWriter

root = "D:/Downloads/mini-imagenet"
train_dataset = MiniImagenet(root, mode='train', batchsz=8, n_way=5, k_shot=1, k_query=15, resize=84)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataset = MiniImagenet(root, mode='test', batchsz=8, n_way=5, k_shot=1, k_query=15, resize=84)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
base_model = SimpleCNN(n_way=5, n_filter=64).cuda()
maml = MAML(base_model, inner_lr=0.01, inner_steps=5)
meta_optimizer = optim.Adam(maml.model.parameters(), lr=0.001)

# TensorboardX
writer = SummaryWriter('./logs')
# # Training
train_maml(maml, train_loader, test_loader, meta_optimizer, epochs=100000, writer=writer)
# Evaluate 
checkpoint = torch.load('maml_miniimagenet_best.pth')
maml.model.load_state_dict(checkpoint)
avg_test_acc = evaluate_maml(maml, test_loader, epochs=1000, eval_inner_steps=10)
print(f"Test acc: {avg_test_acc*100:.4f}%")