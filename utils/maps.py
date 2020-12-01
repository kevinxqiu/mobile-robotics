import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'global_nav'))

import A_star

class Map():

    def __init__(self):

        self.start = (0,0)
        self.resize_factor = 6 # Resize occupancy grid

    def path(self, goal, verbose = False):
        # Define the start and end goal
        
        occupancy_grid = A_star.get_map('map.jpg',self.resize_factor)
        max_x, max_y = occupancy_grid.shape # Size of the map
        max_val = [max_x,max_y]

        # List of all coordinates in the grid
        x,y = np.mgrid[0:max_x:1, 0:max_y:1]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
        coords = list([(int(x[0]), int(x[1])) for x in pos])

        # Define the heuristic, here = distance to goal ignoring obstacles
        h = np.linalg.norm(pos - goal, axis=-1)
        h = dict(zip(coords, h))

        # Run the A* algorithm
        path, visitedNodes = A_star.A_Star(start, goal, h, coords, occupancy_grid, max_val, movement_type="8N")
        path = np.array(path).reshape(-1, 2).transpose()
        visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()

        return path #return a list of coordinates

        if verbose == True:
            # Displaying the map
            cmap = colors.ListedColormap(['white', 'red']) # Select the colors with which to display obstacles and free cells
            fig_astar, ax_astar = A_star.create_empty_plot(max_val)
            ax_astar.imshow(occupancy_grid, cmap=cmap)
            plt.title("Map : free cells in white, occupied cells in red") #joachim: I erased the ; 

            # Plot the best path found and the list of visited nodes
            ax_astar.scatter(visitedNodes[0], visitedNodes[1], marker="o", color = 'orange')
            ax_astar.plot(path[0], path[1], marker="o", color = 'blue')
            ax_astar.scatter(self.start[0], self.start[1], marker="o", color = 'green', s=200)
            ax_astar.scatter(goal[0], goal[1], marker="o", color = 'purple', s=200)