# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:53:48 2020

@author: Celinna

Purpose: Local navigation

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2
import A_star

def move(serial, l_speed=500, r_speed=500, verbose=False):
    """
    Sets the motor speeds of the Thymio 
    param l_speed: left motor speed
    param r_speed: right motor speed
    param verbose: whether to print status messages or not
    param th: thymio serial connection handle
    """
    # Printing the speeds if requested
    if verbose: print("\t\t Setting speed : ", l_speed, r_speed)
    
    # Changing negative values to the expected ones with the bitwise complement
    l_speed = l_speed if l_speed>=0 else 2**16+l_speed
    r_speed = r_speed if r_speed>=0 else 2**16+r_speed

    # Setting the motor speeds
    serial.set_var("motor.left.target", l_speed)
    serial.set_var("motor.right.target", r_speed)
    
    
'''
def go_straight(motor_speed=100, white_threshold=500, verbose=False):
    """
    Go Straight Behaviour of the FSM 
    param motor_speed: the Thymio's motor speed
    param white_threshold: threshold starting which it is considered that the ground sensor saw white
    param verbose: whether to print status messages or not
    """
    if verbose: print("Starting go straight behaviour")
    
    # Move forward, i.e. set motor speeds
    move(l_speed=motor_speed, r_speed=motor_speed)
    
    # Until one of the ground sensors sees some white
    saw_white = False
    
    while not saw_white:
        if test_ground_white(white_threshold, verbose=verbose):
            saw_white=True
            if verbose: print("\t Saw white on the ground, exiting go straight behaviour")
      
    return 
'''


def path_following(verbose=False):
    """
    Follows the planned path
    
    Parameters
    ----------
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    saw_obstacle = False
    
    while not saw_obstacle:
        if test_saw_wall(wall_threshold=500,verbose=verbose):
            saw_obstacle = True
            # Start following wall of obstacle
            wall_following(motor_speed=20,wall_threshold=500,verbose=verbose)
            
    return
    

def test_found_path(verbose=False):
    """
    Test if the robot has returned to its original planned path
    Parameters
    ----------
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    return False
    


def test_saw_wall(serial,  wall_threshold, verbose=False):
    """
    Tests whether one of the proximity sensors saw a wall
    param wall_threshold: threshold starting which it is considered that the sensor saw a wall
    param verbose: whether to print status messages or not
    """
    if any([x>wall_threshold for x in serial['prox.horizontal'][:-2]]):
        if verbose: print("\t\t Saw a wall")
        return True
    
    return False


def wall_following(motor_speed=20, wall_threshold=500, verbose=False):
    """
    Wall following behaviour of the FSM
    param motor_speed: the Thymio's motor speed
    param wall_threshold: threshold starting which it is considered that the sensor saw a wall
    param white_threshold: threshold starting which it is considered that the ground sensor saw white
    param verbose: whether to print status messages or not
    """
    
    if verbose: print("Starting wall following behaviour")
    found_path = False
    
    if verbose: print("\t Moving forward")
    move(l_speed=motor_speed, r_speed=motor_speed)
           
    prev_state="forward"
    
    while not found_path:
        
        if test_saw_wall(wall_threshold, verbose=False):
            if prev_state=="forward": 
                if verbose: print("\tSaw wall, turning clockwise")
                move(l_speed=motor_speed, r_speed=-motor_speed)
                prev_state="turning"
        
        else:
            if prev_state=="turning": 
                if verbose: print("\t Moving forward")
                move(l_speed=motor_speed, r_speed=motor_speed)
                prev_state="forward"

        if test_found_path(verbose): 
            found_path = True

    return 

    
def g_path_FSM(speed, verbose=True):
    while True:
        # Step 1: line following
        path_following(speed, verbose=verbose) # INSERT JOACHIM'S CODE
        
        # Step 2: wall following
        wall_following(speed, verbose=verbose)


# Define the start and end goal
start = (0,0)
goal = (20,40)

 
resize_factor = 6 # Resize occupancy grid
occupancy_grid = A_star.get_map('map.jpg',resize_factor)
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

# Displaying the map
cmap = colors.ListedColormap(['white', 'red']) # Select the colors with which to display obstacles and free cells
fig_astar, ax_astar = A_star.create_empty_plot(max_val)
ax_astar.imshow(occupancy_grid, cmap=cmap)
plt.title("Map : free cells in white, occupied cells in red");

# Plot the best path found and the list of visited nodes
ax_astar.scatter(visitedNodes[0], visitedNodes[1], marker="o", color = 'orange');
ax_astar.plot(path[0], path[1], marker="o", color = 'blue');
ax_astar.scatter(start[0], start[1], marker="o", color = 'green', s=200);
ax_astar.scatter(goal[0], goal[1], marker="o", color = 'purple', s=200);