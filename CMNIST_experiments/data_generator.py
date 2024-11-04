import numpy as np

def generate_distinct_colors(num_colors=5, min_distance=100):
    colors = []

    def l2_distance(color1, color2):
        return np.linalg.norm(np.array(color1) - np.array(color2))

    while len(colors) < num_colors:
        candidate = np.random.randint(0, 256, 3)  # Generate a random RGB color
        if all(l2_distance(candidate, color) >= min_distance for color in colors):
            colors.append(candidate)

    return np.array(colors)

colors = generate_distinct_colors()
print("Generated Colors:", colors)
