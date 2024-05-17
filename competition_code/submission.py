"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np
import math
from time import sleep

# this fuction takes an angle and normalizes it to be between -pi and pi
# in other words, -180 and 180 degrees
def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

# this says, give me 3 things
# (i) a location, expressed as an array
# (ii) a current index, I suspect this means an entry into the waypoints
# (iii) a set of wayopints
# and I'll give you the closest waypoint
def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx



class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.steer_pid_controller = SteeringPIDController()
        self.throttle_pid_controller = ThrottlePIDController()
        self.f = open('race.txt', 'w')
        self.target_waypoint_idx = 5
        self.num_steps = 0
    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        self.prior_speed_error = 0
        self.prior_steer_error = -2
        self.prior_steer_integral = 0
        self.target_waypoint_idx = 5

    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)

        # debugging controls
        show_location=False
        show_velocity=True
        show_steer=False
        write_step=False
        slowmo_start = 1365
        slowmo_end = 1350
        self.num_steps += 1

        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        if (show_location):
            print(f'Vehicle_location: [{vehicle_location[0]:.2f}, {vehicle_location[1]:.2f}, {vehicle_location[2]:.2f}]  ')
            print(f'Closest waypoint is {self.current_waypoint_idx} at',
            f'[{self.maneuverable_waypoints[self.current_waypoint_idx].location[0]:.2f},', 
            f'{self.maneuverable_waypoints[self.current_waypoint_idx].location[1]:.2f},',
            f'{self.maneuverable_waypoints[self.current_waypoint_idx].location[2]:.2f}]')

        ## Step 1: find the target line and the angle to the target line
        target_line, self.target_waypoint_idx = self.get_target_line(vehicle_location, self.current_waypoint_idx, self.target_waypoint_idx, self.maneuverable_waypoints)
        target_steer = np.arctan2(target_line[1],target_line[0])

        ## Step 2: set the steering control to go to the target line

        # Set up the variables needed for the steering PID control
        # vehicle_rotation is roll, pitch, yaw.  we just want the yaw so we use [2]
        current_steer = vehicle_rotation[2]
        steer_error = normalize_rad(target_steer - current_steer) / np.pi
        steer_control = self.steer_pid_controller.run(steer_error, vehicle_velocity_norm, current_steer, target_steer, 
            self.maneuverable_waypoints[self.current_waypoint_idx], self.current_waypoint_idx)

        # print steer
        if (show_steer):
            print(f"Error={steer_error:.3F}, Steer={steer_control:.3F}", 
                  f"Velocity={vehicle_velocity_norm:.3F}, Waypoint={self.current_waypoint_idx:}")

        # slow down in slowmo range to see what is going on
        if (slowmo_start <= self.current_waypoint_idx <= slowmo_end): sleep(0.25)

        # Step 3: Pick a target speed
        current_speed = vehicle_velocity_norm
        target_speed = self.get_target_speed(vehicle_location, self.current_waypoint_idx, self.target_waypoint_idx, self.maneuverable_waypoints)

        # Step 4: Set the throttle
        speed_error = target_speed - current_speed
        throttle_control = self.throttle_pid_controller.run(target_speed, speed_error, vehicle_velocity_norm)

        if (show_velocity):
            print(f"Throttle/Velocity= {throttle_control:.3F}, {vehicle_velocity_norm:.3F}, Waypoint={self.current_waypoint_idx:},",
                  f"Target={self.target_waypoint_idx}, Distance={(self.target_waypoint_idx - self.current_waypoint_idx)%len(self.maneuverable_waypoints)}")

        if (write_step):
            self.f.write(f'{self.num_steps}, ')
            self.f.write(f'Location, {vehicle_location[0]:.2f}, {vehicle_location[1]:.2f}, ')
            self.f.write(f'Velocity, {vehicle_velocity_norm:.2f}, {throttle_control:.3f}, ')
            self.f.write(f'Waypoint, {self.current_waypoint_idx}, {self.maneuverable_waypoints[self.current_waypoint_idx].location[0]:.2f}, ')
            self.f.write(f'{self.maneuverable_waypoints[self.current_waypoint_idx].location[1]:.2f}, ')
            self.f.write(f'Angles, {math.degrees(self.maneuverable_waypoints[self.current_waypoint_idx].roll_pitch_yaw[2]):.1f}, ')
            self.f.write(f'{math.degrees(vehicle_rotation[2]):.1f}, {self.target_waypoint_idx}, {math.degrees(target_steer):.1F}, ')
            self.f.write(f'{math.degrees(normalize_rad(target_steer - current_steer)):.1f}, ')
            self.f.write(f'{steer_error:.2F}, ')
            self.f.write(f'{throttle_control:.2F} ')
            self.f.write('\n')


        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(-throttle_control, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return 
    
    # below are the functions for getting the target line and speed, and the classes for the two PID Controllers

    def get_target_line(self, vehicle_location: np.ndarray, current_idx: int, target_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]):

        # multiple solutions are coded below.  The variable algo determines which solution to use
        # debug controls whether messages are displayed as it is working
        algo = 2
        debug = False

        if (algo==1):
            # Basic version that simply looks 5 waypoints ahead
            # it then we calculate the vector to that waypoint. Note the [:2] gets rid of the z coordinate
            waypoint_to_follow = waypoints[(current_idx + 5) % len(waypoints)]
            return (waypoint_to_follow.location - vehicle_location)[:2], (current_idx + 5) % len(waypoints)

        elif (algo==2):
            # This version looks for a waypoint as far ahead as possible between both a minimum and maximum value, taking into 
            # account three things:
            # 1) The line to that waypoint does not hit a wall, so the angle to the line does not overlap with the angles to the
            # left and right walls of all prior waypoints
            # 2) Once it locks in on a waypoint, it should never pick a closer waypoint, as that can cause odd steering angles
            # 3) if we're going through an S-curve, then don't go past the inflection point in the S-curve where it changes direction
            # Future potential improvements
            # - widen safety to 3 or 4 as steering gets better
            # - increase max_lookahead to 100 or more
            # - figure out how to take curves wider, for example as you approach a sharp right turn, start from the left side
            #   of the track, cut the right wall, and keep going until you are close again to the left side and then straighten out

            # parameters to tweak
            min_lookahead = 5
            max_lookahead = 100
            safety = 2

            # initialize variables
            min_left_angle = 100
            min_right_angle = -100
            waypoint_to_follow = waypoints[target_idx]

            # now see if we can advance our line of sight further
            start_waypoint = (current_idx+min_lookahead)%len(waypoints)
            end_waypoint = (current_idx+max_lookahead)%len(waypoints)
            if (start_waypoint > end_waypoint): 
                end_waypoint += len(waypoints)
            look_ahead = start_waypoint
            if (debug): print(f'analzying from {start_waypoint} to {end_waypoint} out of {len(waypoints)}')

            for i in range(start_waypoint, end_waypoint):
                if (debug): print(f'iterating on {i}')

                # test the next waypoint by drawing a vector to the waypoint, computing the angle to that waypoint.
                there = waypoints[(i) % len(waypoints)]
                vector = (there.location - vehicle_location)[:2]
                angle_to_waypoint = np.arctan2(vector[1],vector[0])

                # Then calculating the angle to the right and left walls at that waypoint.  Since the track curves,
                # we want the narrowest path
                right_wall_angle = angle_to_waypoint - np.arctan2(safety, np.linalg.norm(there.location[:2] - vehicle_location[:2]) )
                left_wall_angle = angle_to_waypoint + np.arctan2(safety, np.linalg.norm(there.location[:2] - vehicle_location[:2]) )
                if (right_wall_angle > min_right_angle):
                    min_right_angle = right_wall_angle
                if (left_wall_angle < min_left_angle):
                    min_left_angle = left_wall_angle

                if (debug) and (0 <= current_idx <= 10):
                    print(f'current={current_idx} ({vehicle_location[0]:.2F},{vehicle_location[1]:.2F}), ',
                        f'target={target_idx}, ' ,
                        f'{i} ({there.location[0]:.2F}, {there.location[1]:.2F})), ' ,
                        f'{math.degrees(angle_to_waypoint):.1F}, ',
                        f'{math.degrees(left_wall_angle):.1F}, ',
                        f'{math.degrees(right_wall_angle):.1F}, ',
                        f'{math.degrees(min_left_angle):.1F}, ',
                        f'{math.degrees(min_right_angle):.1F}')

                # if angle to the waypoint hits a wall, then stop
                if (angle_to_waypoint < min_right_angle or angle_to_waypoint > min_left_angle):
                    break

                # if direction is changed, then stop.  -1 means right, 1 means left, 0 means neutral
                if (i==start_waypoint):
                    previous_angle = angle_to_waypoint
                    direction = 0
                else:
                    if (angle_to_waypoint < previous_angle) and (direction==0):
                        direction = -1
                    elif (angle_to_waypoint < previous_angle) and (direction==1):
                        break
                    elif (angle_to_waypoint > previous_angle) and (direction==0):
                        direction = 1
                    elif (angle_to_waypoint > previous_angle) and (direction==-1):
                        break

                look_ahead = (i) % len(waypoints)
                waypoint_to_follow = there

        # adjust so we never go backwards.  Need to adjust for the fact that after 2774 waypoints go back to 0
        look_ahead_adjusted = (look_ahead + 1000)%len(waypoints)
        target_idx_adjusted = (target_idx + 1000)%len(waypoints)
        if (target_idx_adjusted > look_ahead_adjusted):
            look_ahead = max(look_ahead, target_idx)
        
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        return vector_to_waypoint, look_ahead%len(waypoints)


    # this current version simply sets the speed depending on where you are on the track
    def get_target_speed(self, vehicle_location: np.ndarray, current_idx: int, target_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]):

        # normalize waypoint
        waypoint = self.current_waypoint_idx%len(waypoints) 
        algo = 2

        if (algo==1):
            # set speed depending on where we are on the track
            if (240 < waypoint < 400):  # straightaway 
                target_speed = 75
            elif (450 < waypoint < 540):  # slight curve
                target_speed = 75
            elif (900 < waypoint < 1200):
                target_speed = 35
            elif (1300 < waypoint < 1500):  # sharp S curve
                target_speed = 20
            elif (1500 < waypoint < 1800):  # straightaway
                target_speed = 20
            elif (2100 < waypoint < 2500):
                target_speed = 20
            else:
                target_speed = 50

        elif (algo==2):
            distance = (target_idx-current_idx)%len(waypoints)
            if (distance > 50):
                target_speed = 300
            elif (distance > 40):
                target_speed = 65
            elif (distance > 35):
                target_speed = 45
            elif (distance > 30):
                target_speed = 35
            elif (distance > 20):
                target_speed = 25
            elif (distance > 10):
                target_speed = 20
            else:
                target_speed = 20

            # special handling for the two sharp S curve until we learn how fast the car can slow down
            if (2600 < waypoint < 2800) or (1360 < waypoint < 1450):
                target_speed = 15

        return target_speed
    


# PID controller to set the steering control given an error
class SteeringPIDController():
    def __init__(self):
        self.steer_integral = 0
        self.prior_steer_error = -2
        self.prior_prior_steer_error = -2
        self.prior_steer_integral = 0
        self.maxerror = -10.0
        self.minerror = 10.0
        self.maxovershoot = 0.0
        self.outputed = False
        self.logresults = False

        if (self.logresults):
            self.f = open('steeringPID.txt', 'w')
            self.f.write('Waypoint, ')
            self.f.write('Velocity, ')
            self.f.write('Waypoint Angle, ')
            self.f.write('Target Steer, ')
            self.f.write('Current Steer, ')
            self.f.write('Steer Error, ')
            self.f.write('Steer Control, ')
            self.f.write('PID, ')
            self.f.write('Steer Error, Steer Integral, Steer Deriv, ')
            self.f.write('KP, KI, KD')
            self.f.write('\n')

    def run(self, steer_error:float, velocity:float, current_steer, target_steer, waypoint:roar_py_interface.RoarPyWaypoint, current_waypoint_idx) -> float:
        steer_derivative = 0
        steer_kp = 0
        steer_kd = 0
        steer_ki = 0
        steer_control = float(0)
        if (self.prior_steer_error) != -2:
            steer_derivative = steer_error - self.prior_steer_error
            self.steer_integral += steer_error

        # multiple solutions are coded below.  The variable algo determines which solution to use
        # debug controls whether messages are displayed as it is working
        algo = 2
        debug = False

        # set kp, kd, and ki
        if (algo == 1):
            steer_kp = 4.0 / np.sqrt(velocity)
            steer_kd = 8.0 / (velocity/2)
            steer_ki = 0.002
        else:
            # values determined through trial and error
            if (velocity < 10):
                steer_kp = 1.25
                steer_kd = 3
            elif (velocity < 20):
                steer_kp = 3.0
                steer_kd = 0.5
            elif (velocity < 30):
                steer_kp = 4.0
                steer_kd = 1.0
            else:
                steer_kp = 4
                steer_kd = 1.0
            steer_ki = 0.000

        if (velocity <= 1e-2):
            steer_control = -np.sign(steer_error)
        else:
            steer_control = -(steer_kp * steer_error) - (steer_ki * self.steer_integral) - (steer_kd * steer_derivative) 

        # debug code
        if (self.logresults):
            xP = (steer_kp * steer_error)
            xI = (steer_ki * self.steer_integral) 
            xD = (steer_kd * steer_derivative) 
            self.f.write(f'{current_waypoint_idx:.2F}, ')
            self.f.write(f'{velocity:.2F}, ')
            self.f.write(f'{math.degrees(waypoint.roll_pitch_yaw[2]):.1F}, ')
            self.f.write(f'{math.degrees(target_steer):.1F}, ')
            self.f.write(f'{math.degrees(current_steer):.1F}, ')
            self.f.write(f'{steer_error:.3F}, ')
            self.f.write(f'{steer_control:.3F}, ')
            self.f.write(f'PID=({xP:.3F}/{xI:.3F}/{xD:.3F}), ')
            self.f.write(f'{steer_error:.3F}, {self.steer_integral:.3F}, {steer_derivative:.3F}, ')
            self.f.write(f'{steer_kp:.2F}, {steer_ki:.3F}, {steer_kd:.2F} ')
            self.f.write('\n')

        # debug code for testing kp and kd combinations
        if (debug):
            if (current_waypoint_idx != 2774 and steer_error > self.maxerror): self.maxerror = steer_error
            if (current_waypoint_idx != 2774 and steer_error < self.minerror): self.minerror = steer_error
            if (self.prior_prior_steer_error != -2):
                if ((steer_error > self.prior_steer_error) and (self.prior_steer_error < self.prior_prior_steer_error)) \
                    or ((steer_error < self.prior_steer_error) and (self.prior_steer_error > self.prior_prior_steer_error)) :
                    if (self.maxovershoot < abs(steer_error - self.prior_steer_error)):
                        self.maxovershoot = abs(steer_error - self.prior_steer_error)

            if (current_waypoint_idx == 500) and (not self.outputed):
                with open("testresults.txt", "a") as f:
                    f.write(f'{velocity:.2F}, ')
                    f.write(f'{steer_kp:.2F}, {steer_kd:.2F}, ')
                    f.write(f'{steer_error:.3F}, {self.minerror:.3F}, {self.maxerror:.3F}, ')
                    f.write(f'{self.maxovershoot:.3F}, {current_waypoint_idx:}')
                    f.write('\n')
                self.outputed = True

        self.prior_prior_steer_error = self.prior_steer_error
        self.prior_steer_error = steer_error
        self.prior_steer_integral = self.steer_integral

        return np.clip(steer_control, -1.0, 1.0) 


# PID controller to set the throttle control given an error
class ThrottlePIDController():
    def __init__(self):
        self.prior_speed_error = 0
        self.prior_speed_integral = 0

    def run(self, target_speed:float, speed_error:float, velocity:float) -> float:
        speed_kp = 0.4
        speed_kd = 3
        algo = 2
        throttle_control = 0
        if algo == 1:
            throttle_control = (speed_kp * speed_error) + (speed_kd * (speed_error - self.prior_speed_error))
        if algo == 2:
            target_velocity=target_speed
            if velocity < target_velocity:
                throttle_control = 3*(target_velocity-velocity)/target_velocity
            elif velocity > target_velocity:
                throttle_control = 3*(target_velocity-velocity)/target_velocity
            else:
                throttle_control = 0
        
        self.prior_speed_error = speed_error
        
        return np.clip(throttle_control, -1.0, 1.0) 
