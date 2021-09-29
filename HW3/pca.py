import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy as np

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

  # Filter values that meet threshold
  for i, val in enumerate(vals):
    if (val / val_sum) > perc:
      filt_vals.append((val, i))

  # Sort in desc order
  filt_vals.sort(key = lambda x: x[0], reverse=True)

  # Sort columns
  indices = [filt_val[1] for filt_val in filt_vals]
  vecs = vecs[:,indices]

  # Create diagonal matrix
  arr = np.array([filt_val[0] for filt_val in filt_vals])
  diag = np.diag(arr)

  return diag, vecs

def project_image(img, U):
  projection = None
  for j in range(0, U.shape[1]):
    au = np.dot(np.dot(U[:,j], img), U[:,j])
    if projection is None:
      projection = au
    else:
      projection += au
  return projection

def display_image(orig, proj):

  # Resize images
  orig_res = orig.reshape(32, 32)
  proj_res = proj.reshape(32, 32)

  # Create plots
  fig, (orig_ax, proj_ax) = plt.subplots(1, 2)

  # Set titles
  orig_ax.set_title("Original")
  proj_ax.set_title("Projection")
  
  # Display images
  orig_im = orig_ax.imshow(orig_res.T, aspect='equal')
  proj_im = proj_ax.imshow(proj_res.T, aspect='equal')

  # Add colorbars
  fig.colorbar(orig_im, ax=orig_ax)
  fig.colorbar(proj_im, ax=proj_ax)
  
  # Show
  plt.show()
