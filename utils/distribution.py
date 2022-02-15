import numpy as np

def cosine_similarity(x, y):
    res = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return res