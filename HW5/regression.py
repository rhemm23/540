from matplotlib import pyplot as plt
import numpy as np
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
  return gradients

def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    pass


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    betas = None
    mse = None
    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    result = None
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    return None, None


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph

if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()

dataset = get_dataset('./bodyfat.csv')
print(gradient_descent(dataset, cols=[1,4], betas=[0,0,0]))
