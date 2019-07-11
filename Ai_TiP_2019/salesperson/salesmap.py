import numpy as np
from matplotlib import collections  as mc
import matplotlib.pyplot as plt
from numpy import random, array, load


__all__ = ['SalesMap']

class SalesMap(object):
    """
    Attributes:
    `n`: number of cities
    `coordinates`: (n x 2) array of city coordinates
    """
    def __init__(self, map_name):
        self.coordinates = load(map_name) # coordinates of cities
        self.n = self.coordinates.shape[0] # number of cities.


    def loss(self, path):
        """Takes an ordered list of city indexes and returns length of path."""
        coords = self.coordinates
        path = array(path)
        from_cities = coords[path[:-1]] # all but last city.
        to_cities = coords[path[1:]] # all but first city.
        distance = ((((from_cities - to_cities)**2).sum(axis=1))**.5).sum()
        return distance

    def show(self, path):
        """Takes an ordered list of city indexes and displays the path."""
        lines = []
        for i in range(len(path)-1): # for each edge
            i_city1 = path[i] # index of city
            i_city2 = path[i+1]
            c_city1 = self.coordinates[i_city1]
            c_city2 = self.coordinates[i_city2]
            lines += [(c_city1, c_city2)]
            
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        
        lc = mc.LineCollection(lines, linewidths=2) # draw edges
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        plt.plot(x, y, 'ro')
        plt.show()
        
        


