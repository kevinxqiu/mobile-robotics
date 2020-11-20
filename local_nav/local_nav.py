# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:53:48 2020

@author: Celinna

Purpose: Local navigation

"""

def move(l_speed=500, r_speed=500, verbose=False,th):
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
    th.set_var("motor.left.target", l_speed)
    th.set_var("motor.right.target", r_speed)

if test_functions:
    move(l_speed=100, r_speed=100)
    time.sleep(2)
    move(l_speed=0, r_speed=0)
    
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

if test_functions:
    go_straight()
    move(l_speed=0, r_speed=0)   
    
    
def wall_following(motor_speed=20, wall_threshold=500, white_threshold=200, verbose=False):
    """
    Wall following behaviour of the FSM
    param motor_speed: the Thymio's motor speed
    param wall_threshold: threshold starting which it is considered that the sensor saw a wall
    param white_threshold: threshold starting which it is considered that the ground sensor saw white
    param verbose: whether to print status messages or not
    """
    
    if verbose: print("Starting wall following behaviour")
    saw_black = False
    
    if verbose: print("\t Moving forward")
    move(l_speed=motor_speed, r_speed=motor_speed)
           
    prev_state="forward"
    
    while not saw_black:
        
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

        if test_saw_black(white_threshold, verbose): saw_black = True

    return 

if test_functions:
    wall_following(verbose=True)
    move(l_speed=0, r_speed=0)
    
    
