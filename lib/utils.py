import numpy as np

def randargmax(x):
    """Argmax operator that breaks ties uniformly at random.

    Args:
        x (ndarray): Input array with ndim=1.

    Returns:
        int: Index at which x is maximum. If there are multiple maxima, then one of the indices is chosen uniformly at random.

    """
    mxs = np.amax(x)
    idxs = np.where(x == mxs)[0]
    if len(idxs) > 1:
        return np.random.choice(idxs)
    else:
        return np.argmax(x)
