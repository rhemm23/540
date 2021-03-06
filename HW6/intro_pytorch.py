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
    print(fmt.format(epoch, correct, total, 100 * correct / total, sum_loss / len(train_loader.dataset)))

def evaluate_model(model, test_loader, criterion, show_loss = True):
  model.eval()
  sum_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for data in test_loader:
      inputs, labels = data
      outputs = model.forward(inputs)
      if show_loss:
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  if show_loss:
    avg_loss = sum_loss / len(test_loader.dataset)
    print('Average loss: {0:.4f}'.format(avg_loss))
  print('Accuracy: {0:.2f}%'.format(100 * correct / total))

def predict_label(model, test_images, index):
  output = model.forward(test_images[index])
  probs = F.softmax(output, dim=1)
  named = [
    (probs[0][0], 'zero'),
    (probs[0][1], 'one'),
    (probs[0][2], 'two'),
    (probs[0][3], 'three'),
    (probs[0][4], 'four'),
    (probs[0][5], 'five'),
    (probs[0][6], 'six'),
    (probs[0][7], 'seven'),
    (probs[0][8], 'eight'),
    (probs[0][9], 'nine')
  ]
  named.sort(key = lambda x: x[0], reverse=True)
  for i in range(3):
    print('{0}: {1:.2f}%'.format(named[i][1], named[i][0] * 100))

if __name__ == '__main__':
  '''
  Feel free to write your own test code here to exaime the correctness of your functions. 
  Note that this part will not be graded.
  '''
  criterion = nn.CrossEntropyLoss()
  data_loader = get_data_loader()
  model = build_model()
  train_model(model, data_loader, criterion, 5)
  test_loader = get_data_loader(False)
  data = [test_loader.dataset[0][0], test_loader.dataset[1][0]]
  evaluate_model(model, test_loader, criterion, True)
  predict_label(model, data, 1)
