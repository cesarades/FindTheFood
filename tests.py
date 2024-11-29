
import numpy as np
from utils import combine_masks

result = combine_masks(
[.2, .2, .2, 1], # scores
[np.array([ # mask 1
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 1, 1, 0]]),
np.array([ # mask 2
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 1]]),
np.array([ # mask 3
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0]]),
np.array([ # mask 4
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]])],
[4, 6, 6, 15]).tolist() # ingredients

assert result == [[15, 0, 0, 0], [6, 6, 6, 6], [0, 4, 4, 15]]