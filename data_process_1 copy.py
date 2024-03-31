import csv
import os
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.database.maps_db.imapsdb import IMapsDB
from nuplan.database.maps_db.layer import MapLayer

class DataProcessor:
    def __init__(self, scenarios, maps_db: IMapsDB):
        self.scenarios = scenarios
        self.maps_db = maps_db

    def process_scenario(self, scenario: AbstractScenario):
        csv_data = []
        history_buffer = SimulationHistoryBuffer.initialize_from_scenario(scenario)

        for sample in history_buffer.data:
            # ego vehicle data
            ego_state = sample.ego_state
            ego_pose = self._extract_pose(ego_state)
            csv_data.append(
                self._extract_vehicle_data(ego_state, ego_pose, sample.iteration, len(history_buffer), "ego"))

            # other vehicles data
            for tracked_object in sample.observation.tracked_objects:
                if tracked_object.is_vehicle:
                    vehicle_state = tracked_object.state
                    vehicle_pose = self._extract_pose(vehicle_state)
                    csv_data.append(
                        self._extract_vehicle_data(vehicle_state, vehicle_pose, sample.iteration, len(history_buffer),
                                                   tracked_object.tracking_id))

        return csv_data

    def _extract_pose(self, state):
        return {
            'x': state.center.x,
            'y': state.center.y,
            'z': state.center.z,
            'qw': state.center.heading.real,
            'qx': state.center.heading.imag,
            'qy': state.center.heading.imag,
            'qz': state.center.heading.imag,
            'acceleration': state.acceleration,
            'angular_rate': state.angular_velocity,
            'epsg': state.center.x  # placeholder, need to get the actual EPSG code
        }

    def _extract_vehicle_data(self, vehicle_state, vehicle_pose, iteration, total_iterations, vehicle_id):
        velocity = vehicle_state.velocity
        vehicle_length = vehicle_state.rear_axle_to_front_bumper_distance + vehicle_state.rear_bumper_to_rear_axle_distance
        vehicle_width = vehicle_state.vehicle_parameters.width
        lane_id = self._get_lane_id(vehicle_state.center)

        preceding_vehicle_id, following_vehicle_id, space_headway, time_headway = self._get_headway_info(vehicle_state, lane_id)

        vehicle_data = {
            "Vehicle_ID": vehicle_id,
            "Frame_ID": iteration,
            "Total_Frames": total_iterations,
            "Global_Time": vehicle_state.time_point.time_s,
            "Local_X": vehicle_pose['x'],
            "Local_Y": vehicle_pose['y'],
            "Global_X": vehicle_pose['x'],
            "Global_Y": vehicle_pose['y'],
            "v_length": vehicle_length,
            "v_Width": vehicle_width,
            "v_Class": vehicle_state.vehicle_parameters.vehicle_type,
            "v_Vel": velocity.magnitude(),
            "Lane_ID": lane_id,
            "Preceding": preceding_vehicle_id,
            "Following": following_vehicle_id,
            "Space_Headway": space_headway,
            "Time_Headway": time_headway,
            "qw": vehicle_pose['qw'],
            "qx": vehicle_pose['qx'],
            "qy": vehicle_pose['qy'],
            "qz": vehicle_pose['qz'],
            "Acceleration_x": vehicle_pose['acceleration'].x,
            "Acceleration_y": vehicle_pose['acceleration'].y,
            "Acceleration_z": vehicle_pose['acceleration'].z,
            "Angular_rate_x": vehicle_pose['angular_rate'].x,
            "Angular_rate_y": vehicle_pose['angular_rate'].y,
            "Angular_rate_z": vehicle_pose['angular_rate'].z,
            "EPSG": vehicle_pose['epsg'],
        }

        return vehicle_data

    def _get_lane_id(self, pose):
        lane_segment = self.maps_db.get_lane(pose)
        if lane_segment:
            return lane_segment.lane_id
        return None

    def _get_headway_info(self, vehicle_state, lane_id):
        preceding_vehicle_id = None
        following_vehicle_id = None
        space_headway = None
        time_headway = None

        return preceding_vehicle_id, following_vehicle_id, space_headway, time_headway

    def process_and_save(self, output_path):
        csv_data = []
        for scenario in self.scenarios:
            csv_data.extend(self.process_scenario(scenario))

        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = list(csv_data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

def get_scenario_map():
    scenario_map = {
        'accelerating_at_crosswalk': [15.0, -3.0],
        'accelerating_at_stop_sign': [15.0, -3.0],
        'accelerating_at_stop_sign_no_crosswalk': [15.0, -3.0],
        'accelerating_at_traffic_light': [15.0, -3.0],
        'accelerating_at_traffic_light_with_lead': [15.0, -3.0],
        'accelerating_at_traffic_light_without_lead': [15.0, -3.0],
        'behind_bike': [15.0, -3.0],
        'behind_long_vehicle': [15.0, -3.0],
        'behind_pedestrian_on_driveable': [15.0, -3.0],
        'behind_pedestrian_on_pickup_dropoff': [15.0, -3.0],
        'changing_lane': [15.0, -3.0],
        'changing_lane_to_left': [15.0, -3.0],
        'changing_lane_to_right': [15.0, -3.0],
        'changing_lane_with_lead': [15.0, -3.0],
        'changing_lane_with_trail': [15.0, -3.0],
        'crossed_by_bike': [15.0, -3.0],
        'crossed_by_vehicle': [15.0, -3.0],
        'following_lane_with_lead': [15.0, -3.0],
        'following_lane_with_slow_lead': [15.0, -3.0],
        'following_lane_without_lead': [15.0, -3.0],
        'high_lateral_acceleration': [15.0, -3.0],
        'high_magnitude_jerk': [15.0, -3.0],
        'high_magnitude_speed': [15.0, -3.0],
        'low_magnitude_speed': [15.0, -3.0],
        'medium_magnitude_speed': [15.0, -3.0],
        'near_barrier_on_driveable': [15.0, -3.0],
        'near_construction_zone_sign': [15.0, -3.0],
        'near_high_speed_vehicle': [15.0, -3.0],
        'near_long_vehicle': [15.0, -3.0],
        'near_multiple_bikes': [15.0, -3.0],
        'near_multiple_pedestrians': [15.0, -3.0],
        'near_multiple_vehicles': [15.0, -3.0],
        'near_pedestrian_at_pickup_dropoff': [15.0, -3.0],
        'near_pedestrian_on_crosswalk': [15.0, -3.0],
        'near_pedestrian_on_crosswalk_with_ego': [15.0, -3.0],
        'near_trafficcone_on_driveable': [15.0, -3.0],
        'on_all_way_stop_intersection': [15.0, -3.0],
        'on_carpark': [15.0, -3.0],
        'on_intersection': [15.0, -3.0],
        'on_pickup_dropoff': [15.0, -3.0],
        'on_stopline_crosswalk': [15.0, -3.0],
        'on_stopline_stop_sign': [15.0, -3.0],
        'on_stopline_traffic_light': [15.0, -3.0],
        'on_traffic_light_intersection': [15.0, -3.0],
        'starting_high_speed_turn': [15.0, -3.0],
        'starting_left_turn': [15.0, -3.0],
        'starting_low_speed_turn': [15.0, -3.0],
        'starting_protected_cross_turn': [15.0, -3.0],
        'starting_protected_noncross_turn': [15.0, -3.0],
        'starting_right_turn': [15.0, -3.0],
        'starting_straight_stop_sign_intersection_traversal': [15.0, -3.0],
        'starting_straight_traffic_light_intersection_traversal': [15.0, -3.0],
        'starting_u_turn': [15.0, -3.0],
        'starting_unprotected_cross_turn': [15.0, -3.0],
        'starting_unprotected_noncross_turn': [15.0, -3.0],
        'stationary': [15.0, -3.0],
        'stationary_at_crosswalk': [15.0, -3.0],
        'stationary_at_traffic_light_with_lead': [15.0, -3.0],
        'stationary_at_traffic_light_without_lead': [15.0, -3.0],
        'stationary_in_traffic': [15.0, -3.0],
        'stopping_at_crosswalk': [15.0, -3.0],
        'stopping_at_stop_sign_no_crosswalk': [15.0, -3.0],
        'stopping_at_stop_sign_with_lead': [15.0, -3.0],
        'stopping_at_stop_sign_without_lead': [15.0, -3.0],
        'stopping_at_traffic_light_with_lead': [15.0, -3.0],
        'stopping_at_traffic_light_without_lead': [15.0, -3.0],
        'stopping_with_lead': [15.0, -3.0],
        'traversing_crosswalk': [15.0, -3.0],
        'traversing_intersection': [15.0, -3.0],
        'traversing_narrow_lane': [15.0, -3.0],
        'traversing_pickup_dropoff': [15.0, -3.0],
        'traversing_traffic_light_intersection': [15.0, -3.0],
        'waiting_for_pedestrian_to_cross': [15.0, -3.0]
    }

    return scenario_map

def get_filter_parameters(num_scenarios_per_type=20, limit_total_scenarios=None, shuffle=True):
    # nuplan challenge
    scenario_types = [
        'starting_left_turn',
        'starting_right_turn',
        'starting_straight_traffic_light_intersection_traversal',
        'stopping_with_lead',
        'high_lateral_acceleration',
        'high_magnitude_speed',
        'low_magnitude_speed',
        'traversing_pickup_dropoff',
        'waiting_for_pedestrian_to_cross',
        'behind_long_vehicle',
        'stationary_in_traffic',
        'near_multiple_vehicles',
        'changing_lane',
        'following_lane_with_lead',
    ]

    scenario_tokens = None              # List of scenario tokens to include
    log_names = None                     # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None          # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = False           # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = True         # Whether to remove scenarios where the mission goal is invalid
    shuffle                             # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance


if __name__ == "__main__":
    data_path = "nuplan/dataset/nuplan-v1.1/splits/mini"
    map_path = "nuplan/dataset/maps"
    save_path = "nuplan/processed_data"
    scenarios_per_type = 1
    total_scenarios = None
    shuffle_scenarios = False
    debug = False

    # create save folder
    os.makedirs(save_path, exist_ok=True)

    # get scenarios
    map_version = "nuplan-maps-v1.0"
    sensor_root = None
    db_files = None
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5) # 0.5 means 50% of scenarios
    builder = NuPlanScenarioBuilder(data_path, map_path, sensor_root, db_files, map_version, scenario_mapping=scenario_mapping)
    scenario_filter = ScenarioFilter(*get_filter_parameters(scenarios_per_type, total_scenarios, shuffle_scenarios))
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # get map API
    maps_db = IMapsDB(map_version=map_version, map_root=map_path)

    # process data
    del worker, builder, scenario_filter, scenario_mapping
    processor = DataProcessor(scenarios, maps_db)
    processor.process_and_save(save_path)