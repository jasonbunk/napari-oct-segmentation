from numba import njit
import numpy as np

# Dynamic programming based algorithm.
# Traces an optimal line through an image,
# following lowest valued pixels while obeying the constraint that the line should be smooth (not discontinuous)
# Line "smoothness" is enforced by bounding the slope of the line.
# Such an algorithm is utilized for "seam carving" images, so this code is modified from a seam carving library.

@njit
def get_sloped_seam(energy: np.ndarray, slope:int) -> np.ndarray:
    """Compute the minimum vertical seam from the backward energy map.
    Modified by Jason Bunk from MIT licensed source: https://github.com/li-plus/seam-carving
    Modifications: added slope parameter; sped up.
    """
    h, w = energy.shape
    inf = np.array([np.inf,]*slope, dtype=np.float32)
    cost = np.concatenate((inf, energy[0], inf))
    parent = np.empty((h, w), dtype=np.int32)

    for r in range(1, h):
        nextc = np.copy(cost)
        for jj in range(w):
            mj = 0
            mc = np.inf
            for sj in range(-slope, slope+1):
                if cost[jj+sj+slope] < mc: # argmin
                    mc = cost[jj+sj+slope]
                    mj = sj
            parent[r,jj] = jj+mj
            nextc[slope+jj] = cost[slope+jj+mj] + energy[r,jj]
        cost = nextc

    c = np.argmin(cost[slope:-slope])
    seam = np.empty(h, dtype=np.int32)
    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam
