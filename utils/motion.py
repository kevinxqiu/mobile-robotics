from threading import Timer

class Robot():
    def __init__(self, th):
        self.curr_pos = [0,0,0]
        self.speed = 100
        self.th = th
        self.rt = RepeatedTimer(Ts, self.test_saw_wall) # it auto-starts, no need of rt.start()

    def get_position(self):
        return self.curr_pos
        
    def move_to_target(self, target_pos):
        if target_pos == self.curr_pos[0:2]: return False #if the robot is already at the position or doesn't move
       
        #distance from current position to target
        distance = math.sqrt(((target_pos[0]-self.curr_pos[0])**2)+((target_pos[1]-self.curr_pos[1])**2))
        
        #absolute angle from current position to target (this angle will always be returned between ]-pi;pi])
        path_angle = math.atan2(target_pos[1]-self.curr_pos[1],target_pos[0]-self.curr_pos[0])
        
        #turn angle to get to target relative to Thymio frame
        turn_angle = path_angle - self.curr_pos[2]
        
        print("distance:{}".format(distance))
        print("turn_angle:{}".format(math.degrees(turn_angle)),"\n")
        
        #give commands
        self.turn(turn_angle)
        self.forward(distance)

        #update position and angle of the robot
        self.curr_pos = [target_pos[0],target_pos[1],path_angle]
        
    def turn(self, turn_angle):
        if turn_angle > 0: #turn_angle to the left (180 case will also turn left)
            target_time = (abs(math.degrees(turn_angle)))/38.558 #linear fit model from rad to s for v=100 (change to Kalman)
            left_speed = -self.speed 
            right_speed = self.speed
            
        elif turn_angle < 0: #turn_angle to the right
            target_time = (abs(math.degrees(turn_angle)))/38.558 
            left_speed = self.speed
            right_speed = -self.speed
            
        else: #if turn_angle = 0, do not waste time
            return False
            
        print("target_time_turn:{}".format(target_time))
        t_0 = time.time()
        
        self.move(l_speed=left_speed, r_speed=right_speed)
        
        time.sleep(target_time)
        t_end = time.time()
        print("actual_time_turn:{}".format(t_end-t_0))
        
        #time.sleep(0.1)

    def forward(self, distance): 
        target_time = distance/31.573 #linear fit model from mm to s for v=100 (change to Kalman)
        
        print("target_time_forward:{}".format(target_time))
        t_0 = time.time()
        
        self.move(l_speed=self.speed, r_speed=self.speed)
        
        time.sleep(target_time)
        t_end = time.time()
        print("actual_time_forward:{}".format(t_end-t_0), "\n")
        
        #time.sleep(0.1)
        
    def stop(self):
        move(l_speed=0, r_speed=0)
        
        #time.sleep(0.1)

    def test_saw_wall(self, thread=True, wall_threshold=500, verbose=False):
        
            if any([x>wall_threshold for x in serial['prox.horizontal'][:-2]]):
                if verbose: print("\t\t Saw a wall")
                    if thread: 
                        self.rt.stop() #we stop the thread to not execute test_saw_wall another time
                        # Start following wall of obstacle
                        wall_following(verbose=verbose)
                        self.rt.start(self)
                    else: #we also use test_saw_wall to check if there is STILL a wall (in the wall_folowing function), so we put thread false
                        return True
            return False #to test, not sure we can return smg with the timer, if not, just change the function to return only when thread is false

    def wall_following(self, motor_speed=100, wall_threshold=500, verbose=False):
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
            
            if test_saw_wall(thread= False, wall_threshold, verbose=False):
                if prev_state=="forward": 
                    if verbose: print("\tSaw wall, turning clockwise")
                    self.move(l_speed=motor_speed, r_speed=-motor_speed)
                    prev_state="turning"
            
            else:
                if prev_state=="turning": 
                    if verbose: print("\t Moving forward")
                    self.move(l_speed=motor_speed, r_speed=motor_speed)
                    prev_state="forward"

            if test_found_path(verbose): 
                found_path = True

    def move(self, l_speed=100, r_speed=100, verbose=False):
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
        self.th.set_var("motor.left.target", l_speed)
        self.th.set_var("motor.right.target", r_speed)

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
    
    return False

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