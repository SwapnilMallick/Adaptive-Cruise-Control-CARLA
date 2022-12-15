""" USC 10-DIGIT ID: 2258452509 """
""" Tested in Town06 """

"""This file is the main controller file

Here, you will design the controller for your for the adaptive cruise control system.
"""

from mp1_simulator.simulator import Observation
import numpy as np

# NOTE: Very important that the class name remains the same
class Controller:
    def __init__(self, target_speed: float, distance_threshold: float):
        self.target_speed = target_speed
        self.distance_threshold = distance_threshold

    def run_step(self, obs: Observation, estimate_dist) -> float:
        """This is the main run step of the controller.

        Here, you will have to read in the observatios `obs`, process it, and output an
        acceleration value. The acceleration value must be some value between -10.0 and 10.0.

        Note that the acceleration value is really some control input that is used
        internally to compute the throttle to the car.

        Below is some example code where the car just outputs the control value 10.0
        """

        ego_velocity = obs.velocity
        target_velocity = obs.target_velocity
        dist_to_lead = estimate_dist


        # Do your magic...

        self.target_speed = target_velocity
        #print(f'Actual dist: {obs.distance_to_lead}, Predicted dist: {dist_to_lead}')
        time_gap = 3.3
        d_safe = self.distance_threshold + (time_gap * ego_velocity) + (ego_velocity ** 2 / 600)

        if (dist_to_lead >= d_safe):
            error = self.target_speed - ego_velocity
            acceleration = np.clip((1.5 * error), -10.0, 10.0)

            return acceleration

        else:
            error = 0 - ego_velocity
            deceleration = np.clip((26.0 * error), -10.0, 10.0)

            return deceleration
