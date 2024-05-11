"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

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

        show_location=False
        show_velocity=False
        show_steer=True

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
        target_line = self.get_target_line(vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints)
        target_steer = np.arctan2(target_line[1],target_line[0])

        ## Step 2: set the steering control to go to the target line

        # Set up the variables needed for the steering PID control
        # vehicle_rotation is roll, pitch, yaw.  we just want the yaw so we use [2]
        current_steer = vehicle_rotation[2]
        steer_error = normalize_rad(target_steer - current_steer) / np.pi
        steer_control = self.steer_pid_controller.run(steer_error, vehicle_velocity_norm)

        # print steer
        if (show_steer):
            print(f"Error={steer_error:.3F}, Steer={steer_control:.3F}", 
                  f"Velocity={vehicle_velocity_norm:.3F}, Waypoint={self.current_waypoint_idx:}")


        # Step 3: Pick a target speed
        current_speed = vehicle_velocity_norm
        target_speed = self.get_target_speed(vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints)

        # Step 4: Set the throttle
        speed_error = target_speed - current_speed

        throttle_algo = 3

        if (throttle_algo==1):
            # Option 1 (original): Simple proportional controller to control the vehicle's speed.
            # Seems to crash if target_speed is > 20 m/s, so maxes out at 10.1 m/s
            throttle_control = (speed_kp * speed_error)
        elif (throttle_algo==3):
            throttle_control = self.throttle_pid_controller.run(speed_error, vehicle_velocity_norm)

        if (show_velocity):
            # print(f"Velocity =[{vehicle_velocity[0]:.3f},{vehicle_velocity[1]:.3f},{vehicle_velocity[2]:.3f}]={vehicle_velocity_norm:.3F}")
            print(f"Throttle/Velocity= {throttle_control:.3F}, {vehicle_velocity_norm:.3F}, Waypoint={self.current_waypoint_idx:}")


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

    # this current version simply looks 5 waypoints ahead
    # it then we calculate the vector to that waypoint. Note the [:2] gets rid of the z coordinate
    def get_target_line(self, vehicle_location: np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]):

        waypoint_to_follow = waypoints[(current_idx + 5) % len(waypoints)]
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        return vector_to_waypoint

    # this current version simply sets the speed depending on where you are on the track
    def get_target_speed(self, vehicle_location: np.ndarray, current_idx: int , waypoints : List[roar_py_interface.RoarPyWaypoint]):

        # for now, set speed depending on waypoint so we can at least finish the race
        if (240 < self.current_waypoint_idx < 350):  # straightaway 
            target_speed = 50
        elif (450 < self.current_waypoint_idx < 540):  # slight curve
            target_speed = 20
        elif (900 < self.current_waypoint_idx < 1200):
            target_speed = 50
        elif (1300 < self.current_waypoint_idx < 1500):  # sharp S curve
            target_speed = 20
        elif (1500 < self.current_waypoint_idx < 1800):  # straightaway
            target_speed = 40
        elif (2100 < self.current_waypoint_idx < 2500):
            target_speed = 40
        elif (2600 < self.current_waypoint_idx < 2800):   # sharp S curve
            target_speed = 20
        else:
            target_speed = 20

        return target_speed
    


# PID controller to set the steering control given an error
class SteeringPIDController():
    def __init__(self):
        self.prior_steer_error = -2
        self.prior_steer_integral = 0


    def run(self, steer_error:float, velocity:float) -> float:
        steer_derivative = 0
        steer_integral = 0
        steer_kp = 0
        steer_kd = 0
        steer_ki = 0
        steer_control = float(0)

        if (self.prior_steer_error) != -2:
            steer_derivative = steer_error - self.prior_steer_error
            steer_integral += steer_error
        self.prior_steer_error = steer_error
        self.prior_steer_integral = steer_integral

        # set kp, kd, and ki
        option = 2
        if (option == 1):
            steer_kp = 4.0 / np.sqrt(velocity)
            steer_kd = 8.0 / (velocity/2)
            steer_ki = 0.002
        else:
            if (velocity < 25):
                steer_kp = 0.75
                steer_kd = 1
            elif (velocity < 50):
                steer_kp = 0.5
                steer_kd = 2
            else:
                steer_kp = 0.25
                steer_kd = 0.5
            steer_ki = 0.005

        # debug code:
        # xP = (steer_kp * steer_error)
        # xI = (steer_ki * steer_integral) 
        # xD = (steer_kd * steer_derivative) 
        # print (f"xP={xP:0.2F}, xI={xI:0.2F},xD={xD:0.2F}")

        if (velocity <= 1e-2):
            steer_control = -np.sign(steer_error)
        else:
            steer_control = -(steer_kp * steer_error) - (steer_ki * steer_integral) - (steer_kd * steer_derivative) 

        return np.clip(steer_control, -1.0, 1.0) 


# PID controller to set the throttle control given an error
class ThrottlePIDController():
    def __init__(self):
        self.prior_speed_error = 0
        self.prior_speed_integral = 0

    def run(self, speed_error:float, velocity:float) -> float:
        speed_kp = 0.4
        speed_kd = 3

        throttle_control = (speed_kp * speed_error) + (speed_kd * (speed_error - self.prior_speed_error))

        self.prior_speed_error = speed_error

        return np.clip(throttle_control, -1.0, 1.0) 
