from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
  x = np.load(filename)
  return x - np.mean(x, axis=0)

def get_covariance(dataset):
  res = None
  for x in dataset:
    row_res = np.dot(np.transpose(x), x)
    if res == None:
      res = row_res
    else:
      res += row_res
  return res / (np.shape(dataset)[0] - 1)


def get_eig(S, m):
  print("")


def get_eig_perc(S, perc):
  print("")


def project_image(img, U):
  print("")


def display_image(orig, proj):
  print("")
