import numpy as np


class ShadowPreservor:
    """Keep posterior acoustic shadow contrast from being washed out."""

    def __call__(self, before, after):
        dark = before < np.percentile(before, 20)
        result = after.copy()
        result[dark] = np.minimum(result[dark], before[dark])
        return result
