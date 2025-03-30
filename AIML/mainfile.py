import numpy as np
from scipy.spatial import distance

pokemon_colors = {
    "Red": (255, 0, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Green": (0, 255, 0),
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Brown": (139, 69, 19),
    "Purple": (128, 0, 128),
    "Gray": (128, 128, 128)
}

def getcolor(rgb: np.ndarray) -> str:
    return min(pokemon_colors, key=lambda color: distance.euclidean(rgb, np.array(pokemon_colors[color])))

