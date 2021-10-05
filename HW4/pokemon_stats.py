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

