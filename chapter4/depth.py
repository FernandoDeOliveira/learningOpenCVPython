import numpy as np
from chapter4 import utils


def createMedianMask(disparityMap, validDepthMask, rect=None):
    """Retrun a mask selecting the media layer, plus shadows"""
    if rect is not None:
        x, y, w, h = rect
        disparityMap = disparityMap[y:y+h, x:x+w]
        validDepthMask = validDepthMask[y:y+h, x:x+w]
    median = np.median(disparityMap)
    return np.where((validDepthMask == 0) | (abs(disparityMap - median) < 12), 1.0, 0.0)
