from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
  x = np.load(filename)
  return x - np.mean(x, axis=0)

def get_covariance(dataset):
  res = None
  for x in dataset:
    row = x.reshape(1, -1)
    row_res = np.dot(np.transpose(row), row)
    if res is None:
      res = row_res
    else:
      res += row_res
  return np.divide(res, dataset.shape[0] - 1)

def get_eig(S, m):
  N = S.shape[0]
  vals, vecs = eigh(S, eigvals=(N - m, N - 1))
  return np.diag(np.flip(vals)), np.flip(vecs, axis=1)

def get_eig_perc(S, perc):
  vals, vecs = eigh(S)
  val_sum = np.sum(vals)
  filt_vals = []
  filt_ind = []
  for i, val in enumerate(vals):
    if (val / val_sum) > perc:
      filt_vals.append(val)
      filt_ind.append(i)
  return np.diag(np.flip(np.array(filt_vals))), np.flip(vecs[:,filt_ind], axis=1)

def project_image(img, U):
  print("")


def display_image(orig, proj):
  print("")

x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig_perc(S, 0.07)
print(Lambda)
print(U)
