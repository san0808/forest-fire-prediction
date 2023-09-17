import pandas as pd
import random

# Define areas, soil types, and proximity to water
areas = [
    "Pacific Northwest", "Southern California", "Northeastern United States",
    "Mediterranean", "Southeast Asia", "Central Africa", "South America",
    "Australia", "Canada", "Russia", "Europe", "North Asia", "Africa",
    "Southeast United States", "Southwest United States", "Amazon Rainforest",
    "Central America", "Middle East", "India", "Southeast Africa"
]

soil_types = ["Loamy", "Sandy", "Clayey"]

proximity_to_water = ["Near", "Far"]

# Create a list to store data
data = []

# Generate random data for 1000 rows
for _ in range(1000):
    area = random.choice(areas)
    oxygen = random.uniform(28, 55)
    temperature = random.uniform(18, 40)
    humidity = random.uniform(15, 60)
    wind_speed = random.uniform(5, 20)
    vegetation_density = random.uniform(40, 80)
    water = random.choice(proximity_to_water)
    soil = random.choice(soil_types)
    fire_occurrence = random.randint(0, 1)
    
    data.append([area, oxygen, temperature, humidity, wind_speed, vegetation_density, water, soil, fire_occurrence])

# Create a DataFrame
columns = ["Area", "Oxygen", "Temperature", "Humidity", "Wind Speed", "Vegetation Density", "Proximity to Water", "Soil Type", "Fire Occurrence"]
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
df.to_csv("Synthetic_Forest_fire_dataset.csv", index=False)
