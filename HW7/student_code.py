# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms

class LeNet(nn.Module):

  def __init__(self, input_shape=(32, 32), num_classes=100):

    super(LeNet, self).__init__()

    # Convolutional layers
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)

    # Calculate shape of the output of
    # the second convolutional layer
    conv2_w = (((input_shape[0] - 4) // 2) - 4) // 2
    conv2_h = (((input_shape[0] - 4) // 2) - 4) // 2

    # Linear layers
    self.lin1 = nn.Linear(16 * conv2_h * conv2_w, 256)
    self.lin2 = nn.Linear(256, 128)
    self.lin3 = nn.Linear(128, num_classes)

    # Other layers
    self.maxpool = nn.MaxPool2d(2, stride=2)
    self.flatten = nn.Flatten()
    self.relu = nn.ReLU()

  def forward(self, x):

    # Input size
    N, C, W, H = x.size()

    # Calculate layer shapes
    layer1_shape = [
      N,
      6,
      (W - 4) // 2,
      (H - 4) // 2
    ]
    layer2_shape = [
      N,
      16,
      (layer1_shape[2] - 4) // 2,
      (layer1_shape[3] - 4) // 2
    ]
    layer3_shape = [
      N,
      layer2_shape[1] * layer2_shape[2] * layer2_shape[3]
    ]
    layer4_shape = [
      N,
      256
    ]
    layer5_shape = [
      N,
      128
    ]
    layer6_shape = [
      N,
      100
    ]

    # Setup shape dict
    shape_dict = {
      1: layer1_shape,
      2: layer2_shape,
      3: layer3_shape,
      4: layer4_shape,
      5: layer5_shape,
      6: layer6_shape
    }

    # Apply convolutional layers
    x = self.maxpool(self.relu(self.conv1(x)))
    x = self.maxpool(self.relu(self.conv2(x)))

    # Flatten to a single dimension
    x = self.flatten(x)

    # Apply linear layers
    x = self.relu(self.lin1(x))
    x = self.relu(self.lin2(x))
    x = self.lin3(x)

    return x, shape_dict

def count_model_params():
  model = LeNet()
  total_cnt = 0
  for name, param in model.named_parameters():
    cnt = 1
    dims = param.size()
    for dim in dims:
      cnt *= dim
    total_cnt += cnt
  return total_cnt / 1e6


def train_model(model, train_loader, optimizer, criterion, epoch):
  """
  model (torch.nn.module): The model created to train
  train_loader (pytorch data loader): Training data loader
  optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
  criterion (nn.CrossEntropyLoss) : Loss function used to train the network
  epoch (int): Current epoch number
  """
  model.train()
  train_loss = 0.0
  for input, target in tqdm(train_loader, total=len(train_loader)):
    ###################################
    # fill in the standard training loop of forward pass,
    # backward pass, loss computation and optimizer step
    ###################################

    # 1) zero the parameter gradients
    optimizer.zero_grad()
    # 2) forward + backward + optimize
    output, _ = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Update the train_loss variable
    # .item() detaches the node from the computational graph
    # Uncomment the below line after you fill block 1 and 2
    train_loss += loss.item()

  train_loss /= len(train_loader)
  print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

  return train_loss

def test_model(model, test_loader, epoch):
  model.eval()
  correct = 0
  with torch.no_grad():
    for input, target in test_loader:
      output, _ = model(input)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
  test_acc = correct / len(test_loader.dataset)
  print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
      epoch + 1, 100. * test_acc))
  return test_acc
