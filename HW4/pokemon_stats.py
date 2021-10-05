import matplotlib.pyplot as plt
import numpy as np
import random
import math
import csv

def load_data(filepath):
  pokemon = []
  with open(filepath, 'r') as file:
    reader = csv.DictReader(file)
    for i in range(20):
      row = next(reader)
      pokemon.append({
        '#': int(row['#']),
        'Name': row['Name'],
        'Type 1': row['Type 1'],
        'Type 2': row['Type 2'],
        'Total': int(row['Total']),
        'HP': int(row['HP']),
        'Attack': int(row['Attack']),
        'Defense': int(row['Defense']),
        'Sp. Atk': int(row['Sp. Atk']),
        'Sp. Def': int(row['Sp. Def']),
        'Speed': int(row['Speed'])
      })
  return pokemon

def calculate_x_y(stats):
  x = stats['Attack'] + stats['Sp. Atk'] + stats['Speed']
  y = stats['Defense'] + stats['Sp. Def'] + stats['HP']
  return x, y

def hac(dataset):

  # Init variables
  m = len(dataset)
  Z = np.empty((m - 1, 4))
  clusters = []

  # Build initial clusters
  for i, data_point in enumerate(dataset):
    clusters.append(([data_point], i))

  # Perform m - 1 iterations
  for i in range(m - 1):

    # Setup iteration variables
    min_clust_dist = float('inf')
    min_clust_a = None
    min_clust_b = None

    # Iterate over clusters
    for j in range(len(clusters)):
      for k in range(len(clusters)):
        if j != k:
          clust_dist = float('inf')
          clust_a = clusters[j]
          clust_b = clusters[k]

          # Iterate over each data_point in clusters a and b
          for data_point_a in clust_a[0]:
            for data_point_b in clust_b[0]:

              # Calculate data point dist
              dist = math.sqrt(abs(data_point_a[0] - data_point_b[0])**2 + abs(data_point_a[1] - data_point_b[1])**2)
              if dist < clust_dist:
                clust_dist = dist

          # Determine if we should update min clust
          update_min_clusts = False
          if clust_dist < min_clust_dist:
            update_min_clusts = True
          elif clust_dist == min_clust_dist:
            # Calc cur index values to break ties
            cur_min_ind = min(min_clust_a[1], min_clust_b[1])
            cur_max_ind = max(min_clust_a[1], min_clust_b[1])

            # Calc new index values
            new_min_ind = min(clust_a[1], clust_b[1])
            new_max_ind = max(clust_a[1], clust_b[1])

            # Break ties
            if new_min_ind < cur_min_ind:
              update_min_clusts = True
            elif new_min_ind == cur_min_ind:
              update_min_clusts = new_max_ind < cur_max_ind

          # Update min clust
          if update_min_clusts:
            min_clust_dist = clust_dist
            min_clust_a = clust_a
            min_clust_b = clust_b

    # Store iteration info
    Z[i][0] = min(min_clust_a[1], min_clust_b[1])
    Z[i][1] = max(min_clust_a[1], min_clust_b[1])
    Z[i][2] = min_clust_dist
    Z[i][3] = len(min_clust_a[0]) + len(min_clust_b[0])

    # Merge the two clusters
    merged_clust = (min_clust_a[0] + min_clust_b[0], m + i)

    # Update cluster list
    clusters.append(merged_clust)
    clusters = [clust for clust in clusters if (clust[1] != min_clust_a[1]) and (clust[1] != min_clust_b[1])]
  
  return Z

def random_x_y(m):
  data = []
  for i in range(m):
    data.append((random.randrange(1, 360), random.randrange(1, 360)))
  return data

def imshow_hac(dataset):
  # Init data
  clusts = {}
  m = len(dataset)
  pokemon_stats = []

  # For each pokemon
  for i, pokemon in enumerate(dataset):
    # Calc pokemon stats
    data_point = calculate_x_y(pokemon)
    pokemon_stats.append(data_point)

  Z = hac(pokemon_stats)

  for i in range(m):
    # Generate color
    color = '#'
    for j in range(6):
      color += random.choice('0123456789ABCDEF')

    # Plot initial scatter
    plt.scatter(pokemon_stats[i][0], pokemon_stats[i][1], c=color)
    clusts[i] = ([pokemon_stats[i]], color)

  # Show plot
  plt.show()
  plt.pause(0.1)

  # Iterate over Z
  for i in range(m - 1):
    clust_a = clusts[Z[i][0]]
    clust_b = clusts[Z[i][1]]

    min_dist = float('inf')
    min_pnt_a = None
    min_pnt_b = None

    # Find points that merged
    for data_point_a in clust_a[0]:
      for data_point_b in clust_b[0]:
        dist = math.sqrt(abs(data_point_a[0] - data_point_b[0])**2 + abs(data_point_a[1] - data_point_b[1])**2)
        if (dist < min_dist):
          min_dist = dist
          min_pnt_a = data_point_a
          min_pnt_b = data_point_b

    # Replot each point in merged clust to update color
    merged_clust = (clust_a[0] + clust_b[0], clust_a[1])
    for data_point in merged_clust[0]:
      plt.scatter(data_point[0], data_point[1], merged_clust[1])

    # Plot line between connecting clusters
    plt.plot([min_pnt_a[0], min_pnt_b[0]], [min_pnt_a[0], min_pnt_b[0]], merged_clust[1])
    plt.draw()
    plt.pause(0.1)

    # Delete old clusts
    del clusts[Z[i][0]]
    del clusts[Z[i][0]]

pokemon = load_data("./Pokemon.csv")
imshow_hac(pokemon)
