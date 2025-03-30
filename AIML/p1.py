import pypokedex
from K_means import *
from randomtree import * 
from cnn import *
from vit import *
from AI_part import *
from mainfile import *
import csv
import numpy as np


with open("KMean_Predictions.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Color"])  # Header
    for i in range(721):
        try:
            pokemon = pypokedex.get(dex=i+1)
            pokemon_name = f"{i+1}: {pokemon.name.capitalize()}" 
            print(pokemon_name)
            predicted_color = Model_KMeans(f"pokemon/{i+1}.png")
            writer.writerow([pokemon.name.capitalize(), response(predicted_color)])
            if i%15==0:
                time.sleep(2)
        except:
            pass

print("CSV file created: pokemon_predictions.csv")
