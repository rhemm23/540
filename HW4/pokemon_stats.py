import numpy as np
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
          for data_point_a in clust_a:
            for data_point_b in clust_b:

              # Calculate data point dist
              dist = math.sqrt(abs(data_point_a[0] - data_point_b[0])**2 + abs(data_point_a[1] - data_point_b[1])**2)
              if dist < clust_dist:
                clust_dist = dist

          # Update min distance between all clusters
          if clust_dist < min_clust_dist:
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
    clusters = filter(lambda clust: clust[1] != min_clust_a[1] && clust[1] != min_clust_b[1], clusters)

pokemon = load_data("./Pokemon.csv")
print(calculate_x_y(pokemon[1]))
