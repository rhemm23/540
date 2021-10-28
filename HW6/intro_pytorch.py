import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training = True):
  custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])
  data_set = datasets.MNIST('./data', train=training, download=True, transform=custom_transform)
  return torch.utils.data.DataLoader(data_set, batch_size = 50)

def build_model():
  return nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
  )

def train_model(model, train_loader, criterion, T):
  opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  model.train()
  for epoch in range(T):
    sum_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader):
      inputs, labels = data
      opt.zero_grad()
      outputs = model.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      opt.step()
      sum_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    fmt = 'Train Epoch: {0} Accuracy: {1}/{2}({3:.2f}%) Loss: {4:.3f}'
    print(fmt.format(epoch, correct, total, correct / total, sum_less / total))

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """


if __name__ == '__main__':
  '''
  Feel free to write your own test code here to exaime the correctness of your functions. 
  Note that this part will not be graded.
  '''
  criterion = nn.CrossEntropyLoss()
  data_loader = get_data_loader()
  model = build_model()
  train(model, data_loader, criterion, 5)
