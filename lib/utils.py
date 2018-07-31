import numpy as np

def randargmax(x):
    """
        Argmax operator that breaks ties uniformly at random
    """
    mxs = np.amax(x)
    idxs = np.where(x == mxs)[0]
    if len(idxs) > 0:
        return np.random.choice(idxs)
    else:
        return np.argmax(x)