import numpy as np


class TreeModel(object):
    """Class representing a tree"""

    def __init__(self):
        self.points = None

    def initialize(self, points: np.ndarray):
        """Initialize a tree model

        Parameters
        ----------
        points : numpy.ndarray
            Points of a cluster representing a tree

        """
        self.points = points

    def get_metrics(self):
        """Get tree metrics"""

        pass
