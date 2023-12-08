import pandas as pd

df = pd.read_csv("model/BREEZE_Food_Carbon_Emission.csv")

def getCarbonEmission(name: str):
  """Get Food Recipe and Emission"""
  # Filter the dataframe by name
  filtered_df = df[df['MAKANAN'] == name]

  # Filter columns
  filtered_df = filtered_df[['BAHAN', 'BERAT (kg)', 'Carbon Footprint']]

  ingredients = []
  for _, row in filtered_df.iterrows():
    ingredients.append({
      "bahan": row['BAHAN'],
      "berat": row['BERAT (kg)'],
      "carbon_footprint": row['Carbon Footprint']
    })

  # Calculate total emissions
  total_emissions = filtered_df['Carbon Footprint'].sum()

  return ingredients, total_emissions