from matplotlib import pyplot as plt
import numpy as np
import random
import math
import csv

def get_dataset(filename):
  dataset = []
  with open(filename, 'r') as file:
    data_reader = csv.DictReader(file)
    for row in data_reader:
      dataset.append([
        float(row['BODYFAT']),
        float(row['DENSITY']),
        float(row['AGE']),
        float(row['WEIGHT']),
        float(row['HEIGHT']),
        float(row['ADIPOSITY']),
        float(row['NECK']),
        float(row['CHEST']),
        float(row['ABDOMEN']),
        float(row['HIP']),
        float(row['THIGH']),
        float(row['KNEE']),
        float(row['ANKLE']),
        float(row['BICEPS']),
        float(row['FOREARM']),
        float(row['WRIST'])
      ])
  return np.array(dataset)

def print_stats(dataset, col):
  # Number of data points
  n = dataset.shape[0]
  print(n)

  # Mean
  mean = 0
  for data_point in dataset:
    mean += data_point[col]
  mean *= (1 / n)
  print('{:0.2f}'.format(mean))

  # Standard deviation
  std_dev = 0
  for data_point in dataset:
    std_dev += (mean - data_point[col])**2
  std_dev = math.sqrt(std_dev * (1 / (n - 1)))
  print('{:0.2f}'.format(std_dev))

def regression(dataset, cols, betas):
  mse = 0
  for data_point in dataset:
    point_sum = betas[0]
    for i in range(len(cols)):
      point_sum += data_point[cols[i]] * betas[i + 1]
    mse += (point_sum - data_point[0])**2
  return mse * (1 / dataset.shape[0])

def gradient_descent(dataset, cols, betas):
  # Initial variables
  n = dataset.shape[0]
  gradients, errs = [], []

  for data_point in dataset:
    err_sum = betas[0]
    for i in range(len(cols)):
      err_sum += data_point[cols[i]] * betas[i + 1]
    errs.append(err_sum - data_point[0])
  
  # With respect to beta 0
  gradients.append(sum(errs) * (2 / n))

  # With respect to cols
  for col_index in cols:
    part_sum = 0
    for i, data_point in enumerate(dataset):
      part_sum += errs[i] * data_point[col_index]
    gradients.append(part_sum * (2 / n))
  return np.array(gradients)

def iterate_gradient(dataset, cols, betas, T, eta):
  for i in range(T):
    gradients = gradient_descent(dataset, cols, betas)
    for j in range(len(betas)):
      betas[j] = betas[j] - (eta * gradients[j])
    print('{} '.format(i + 1), end='')
    print('{:0.2f} '.format(regression(dataset, cols, betas)), end='')
    for j in range(len(betas)):
      print('{:0.2f}'.format(betas[j]), end='')
      if j != (len(betas) - 1):
        print(' ', end='')
      else:
        print()

def compute_betas(dataset, cols):
  # Form X and Y from dataset
  X = dataset[:,cols]
  Y = dataset[:,0]

  # Column of ones for beta_0
  n = dataset.shape[0]
  bias_col = np.ones((n, 1))

  # Add ones to beginning of X
  X = np.concatenate((bias_col, X), axis=1)

  # Calculate betas
  betas = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

  # Calculate mse
  mse = regression(dataset, cols, betas)

  return (mse, *betas)

def predict(dataset, cols, features):
  # Compute betas
  betas = compute_betas(dataset, cols)
  
  # Form prediction
  res = betas[1]
  for i in range(len(cols)):
    res += betas[i + 2] * features[i]
  return res

def synthetic_datasets(betas, alphas, X, sigma):
  y_lin = []
  y_quad = []
  for x_i in X:
    # Compute linear data
    z_i_lin = np.random.normal(0, sigma)
    y_lin.append([betas[0] + betas[1] * x_i[0] + z_i_lin, x_i[0]])

    # Compute quadratic data
    z_i_quad = np.random.normal(0, sigma)
    y_quad.append([alphas[0] + alphas[1] * x_i[0] * x_i[0] + z_i_quad, x_i[0]])
  return (np.array(y_lin), np.array(y_quad))

def plot_mse():
  from sys import argv
  if len(argv) == 2 and argv[1] == 'csl':
      import matplotlib
      matplotlib.use('Agg')

  # Random X values
  X = np.random.randint(-100, 101, (1000, 1))

  # Random alpha/beta values
  betas = [random.random() + 1, random.random() + 1]
  alphas = [random.random() + 1, random.random() + 1]

  # Defined sigma values
  sigmas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]

  # MSE values
  lin_mse = []
  quad_mse = []

  # For all sigmas
  for sigma in sigmas:
    # Synthesize data
    syn_data = synthetic_datasets(betas, alphas, X, sigma)

    # Find betas for each dataset
    lin_betas = compute_betas(syn_data[0], cols=[1])
    quad_betas = compute_betas(syn_data[1], cols=[1])

    # Store mse
    lin_mse.append(lin_betas[0])
    quad_mse.append(quad_betas[0])

  # Set scale
  plt.yscale('log')
  plt.xscale('log')

  # Set axis labels
  plt.xlabel('Standard Deviation of Error Term')
  plt.ylabel('MSE of Trained Model')

  # Plot linear and quadratic data
  plt.plot(sigmas, lin_mse, '-o', label='MSE of Linear Dataset')
  plt.plot(sigmas, quad_mse, '-o', label='MSE of Quadratic Dataset')

  # Save figure
  plt.legend()
  plt.savefig('mse.pdf')

if __name__ == '__main__':
  ### DO NOT CHANGE THIS SECTION ###
  plot_mse()
