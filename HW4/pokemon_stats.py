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
  clusters = []
  for i, data_point in enumerate(dataset):
    clusters.append(([data_point], i))
  Z = np.empty((len(dataset), 4))
  for i in range(len(dataset) - 1):
    min_clust_dist = float('inf')
    min_clust_a = None
    min_clust_b = None
    for j in range(len(clusters)):
      for k in range(len(clusters)):
        if j != k:
          clust_dist = float('inf')
          clust_a = clusters[j]
          clust_b = clusters[k]
          for data_point_a in clust_a:
            for data_point_b in clust_b:
              dist = math.sqrt(abs(data_point_a[0] - data_point_b[0])**2 + abs(data_point_a[1] - data_point_b[1])**2)
              if dist < clust_dist:
                clust_dist = dist
          if clust_dist < min_clust_dist:
            min_clust_dist = clust_dist
            min_clust_a = clust_a
            min_clust_b = clust_b


pokemon = load_data("./Pokemon.csv")
print(calculate_x_y(pokemon[1]))
