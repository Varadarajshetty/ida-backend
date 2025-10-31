import numpy as np

def embed_texts(texts):
    out = []
    for t in texts:
        h = abs(hash(t)) % 10000000
        vec = [(h >> (i*8)) & 255 for i in range(32)]
        a = np.array(vec, dtype=float)
        a = a / (np.linalg.norm(a) + 1e-12)
        out.append(a.tolist())
    return out
