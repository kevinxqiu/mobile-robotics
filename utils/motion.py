import os
import sys
import time
import serial
import math
import numpy as np
from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


class Robot():
    def __init__(self, th):
        self.curr_pos = [0,0,0]
        self.wheeltowheel_length = 95 #mm
        self.speed = 100 #Thymio speed
        self.speed_to_mm_s = 0.31573 #multiply with Thymio speed to get mm/s
        #self.speed_to_deg_s = 0.38558 #multiply with Thymio speed to get deg/s
        self.speed_to_deg_s = math.degrees(2*self.speed_to_mm_s/self.wheeltowheel_length)
        self.th = th
        #self.rt_speed = RepeatedTimer(1, self.get_speed) # it auto-starts, no need of rt.start()
        #self.rt = RepeatedTimer(0.4, self.test_saw_wall) # it auto-starts, no need of rt.start()

    def get_position(self):
        return self.curr_pos

    def set_position(self, pos):
        self.curr_pos = pos

    def get_speed(self): #Measures Thymio speed at each wheel
        self.measured_speed = np.array([self.th["motor.left.speed"], self.th["motor.right.speed"]])

        if self.measured_speed[0] > 9999:
            self.measured_speed[0] = self.measured_speed[0] - 2**16
        if self.measured_speed[1] > 9999:
            self.measured_speed[1] = self.measured_speed[1] - 2**16

        #convert to mm/s
        self.measured_speed = self.speed_to_mm_s * self.measured_speed

        #print(self.measured_speed)
        return self.measured_speed

    def set_speed(self, speed):
        self.speed = speed

    def move_to_target(self, target_pos):
        if target_pos == self.curr_pos[0:2]: #if the robot is already at the position or doesn't move
            return False

        #distance from current position to target
        distance = math.sqrt(((target_pos[0] - self.curr_pos[0]) **2) + ((target_pos[1] - self.curr_pos[1]) **2))

        #absolute angle from current position to target (this angle will always be returned between ]-180;180])
        path_angle = math.degrees(math.atan2(target_pos[1] - self.curr_pos[1], target_pos[0] - self.curr_pos[0]))

        #turn angle to get to target (relative to Thymio frame)
        turn_angle = path_angle - self.curr_pos[2]
        if abs(turn_angle) > 180:
             turn_angle = (turn_angle + 360) % 360

        #give commands
        self.turn(turn_angle)
        self.go_straight(distance)

        #update position and angle of the robot
        #self.curr_pos = [target_pos[0],target_pos[1],path_angle]

    def go_straight(self, distance):
        print("distance:{}".format(distance))

        target_time = abs(distance) / (self.speed * self.speed_to_mm_s)  #linear fit model from mm to s for v=100 (change to Kalman)

        #print("target_go_straight:{} s".format(target_time))

        t_0 = time.time()

        if distance > 0: #go forward
            self.move(l_speed=self.speed, r_speed=self.speed)
        elif distance < 0: #go backwards
            self.move(l_speed=-self.speed, r_speed=-self.speed)

        time.sleep(target_time)
        t_end = time.time()

        print("go_straight_took:{} s".format(t_end-t_0), "\n")

        #time.sleep(0.1)

    def turn(self, turn_angle):
        print("turn_angle:{}".format(turn_angle))

        target_time = abs(turn_angle)/ (self.speed * self.speed_to_deg_s) #linear fit model from degrees to s for v=100 (change to Kalman)

        #print("target_turn:{} s".format(target_time))

        t_0 = time.time()

        if turn_angle > 0: #turn_angle to the left
            self.move(l_speed=-self.speed, r_speed=self.speed)
        elif turn_angle < 0: #turn_angle to the right
            self.move(l_speed=self.speed, r_speed=-self.speed)
        else: #if turn_angle = 0, do not waste time
            return False

        time.sleep(target_time)
        t_end = time.time()

        print("actual_turn_took:{} s".format(t_end-t_0), "\n")

        #time.sleep(0.1)

    def move(self, l_speed=100, r_speed=100, verbose=False):
        # Printing the speeds if requested
        if verbose: print("\t\t Setting speed : ", l_speed, r_speed)
        # Changing negative values to the expected ones with the bitwise complement
        if l_speed < 0:
            l_speed = l_speed + 2**16
        if r_speed < 0:
            r_speed = r_speed + 2**16

        # Setting the motor speeds
        self.th.set_var("motor.left.target", l_speed)
        self.th.set_var("motor.right.target", r_speed)

    def stop(self):
        self.move(l_speed=0, r_speed=0)

        #time.sleep(0.1)

    def test_saw_wall(self, thread=True, wall_threshold=500, verbose=True):
        if any([x>wall_threshold for x in self.th['prox.horizontal'][:-2]]):
            if verbose: print("\t\t Saw a wall")
            if thread:
                self.rt.stop() #we stop the thread to not execute test_saw_wall another time
                # Start following wall of obstacle
                self.wall_following(verbose=verbose)
                self.rt.start()
            else: #we also use test_saw_wall to check if there is STILL a wall (in the wall_folowing function), so we put thread false
                return True
        return False #to test, not sure we can return smg with the timer, if not, just change the function to return only when thread is false

    def wall_following(self, motor_speed=100, wall_threshold=500, verbose=True):
        """
        Wall following behaviour of the FSM
        param motor_speed: the Thymio's motor speed
        param wall_threshold: threshold starting which it is considered that the sensor saw a wall
        param white_threshold: threshold starting which it is considered that the ground sensor saw white
        param verbose: whether to print status messages or not
        """

        if verbose: print("Starting wall following behaviour")
        found_path = False
        self.move(l_speed=100, r_speed=100, verbose=False)

        print("fini")
        prev_state="forward"
        count=0
        while not found_path:
            s_w = self.test_saw_wall(thread=False, wall_threshold=wall_threshold)
            print("sw: {}".format(s_w))
            if s_w:
                if prev_state=="forward":
                    if verbose: print("Saw wall move right")
                    self.move(l_speed=100, r_speed=-100, verbose=False) #turn 90 right
                    time.sleep(0.1)
                    prev_state="turning"
                    count=0
            else:
                if prev_state=="turning":
                    if verbose: print("Moving forward")
                    self.move(l_speed=100, r_speed=100, verbose=False)#forward
                    time.sleep(0.5)
                    prev_state="forward"
                else:
                    self.move(l_speed=-100, r_speed=100, verbose=False)#turn 90 left
                    time.sleep(0.15)
                    if verbose: print("Moving left")
                    prev_state="turning"
                count +=1
                print(count)
            if count >= 10:
                found_path = True
                self.stop()

    def test_found_path(self, verbose=False):
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
        for coord in self.path:
            if self.get_position() == coord:
                return True

        return False
