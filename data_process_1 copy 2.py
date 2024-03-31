import csv
import os
import numpy as np
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from common_utils import *

from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.simulation.observation.observation_type import Sensors, DetectionsTracks
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType, TrafficLightStatusData, SemanticMapLayer
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from shapely.geometry import Point
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.agent import AgentState
from nuplan.common.actor_state.agent import Agent



class DataProcessor:
    def __init__(self, scenarios, map_api: AbstractMap):
        self.scenarios = scenarios
        self.map_api = map_api

    def process_scenario(self, scenario: AbstractScenario):
        csv_data = []
        observation_type = DetectionsTracks  # 使用 DetectionsTracks 类作为观测类型
        buffer_size = 10  # 设置缓冲区大小为10

        # 获取过去的观测数据
        past_observations = list(scenario.get_past_tracked_objects(iteration=0, time_horizon=10, num_samples=buffer_size))

        # 获取过去的自车状态
        past_ego_states = list(scenario.get_ego_past_trajectory(iteration=0, time_horizon=10, num_samples=buffer_size))

        # 使用 initialize_from_list 方法创建 SimulationHistoryBuffer
        history_buffer = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=buffer_size,
            ego_states=past_ego_states,
            observations=past_observations,
            sample_interval=scenario.database_interval
        )

        for sample_ego_state, sample_observation in zip(history_buffer.ego_states, history_buffer.observations):
            # ego vehicle data
            ego_pose = self._extract_pose(sample_ego_state)
            csv_data.append(
                self._extract_vehicle_data(sample_ego_state, ego_pose, sample_ego_state.time_point.time_s, buffer_size, "ego", scenario))

            # other vehicles data
            for tracked_object in sample_observation.tracked_objects:
                if tracked_object.tracked_object_type in [TrackedObjectType.VEHICLE, TrackedObjectType.BICYCLE, TrackedObjectType.PEDESTRIAN]:
                    agent = tracked_object
                    vehicle_pose = self._extract_pose(agent)
                    csv_data.append(
                        self._extract_vehicle_data(agent, vehicle_pose, sample_ego_state.time_point.time_s, buffer_size,
                                                tracked_object.track_token, scenario))

        return csv_data

    def _extract_pose(self, state):
        if isinstance(state, EgoState):
            return {
                'x': state.center.x,
                'y': state.center.y,
                'qw': np.cos(state.center.heading / 2),
                'qx': 0,
                'qy': 0,
                'qz': np.sin(state.center.heading / 2),
                'velocity_x': state.dynamic_car_state.center_velocity_2d.x,
                'velocity_y': state.dynamic_car_state.center_velocity_2d.y,
                'acceleration_x': state.dynamic_car_state.center_acceleration_2d.x,
                'acceleration_y': state.dynamic_car_state.center_acceleration_2d.y,
                'angular_velocity': state.dynamic_car_state.angular_velocity,
                'epsg': state.center.x  # placeholder, need to get the actual EPSG code
            }
        elif isinstance(state, Agent):
            return {
                'x': state.box.center.x,
                'y': state.box.center.y,
                'qw': np.cos(state.box.center.heading / 2),
                'qx': 0,
                'qy': 0,
                'qz': np.sin(state.box.center.heading / 2),
                'velocity_x': state.velocity.x,
                'velocity_y': state.velocity.y,
                'acceleration_x': 0,
                'acceleration_y': 0,
                'angular_velocity': state.angular_velocity if state.angular_velocity is not None else 0,
                'epsg': state.box.center.x  # placeholder, need to get the actual EPSG code
            }
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

    def _extract_vehicle_data(self, vehicle_state, vehicle_pose, iteration, total_iterations, vehicle_id, scenario):
        if isinstance(vehicle_state, EgoState):
            # 提取自车速度信息
            velocity = vehicle_state.dynamic_car_state.center_velocity_2d

            # 提取自车尺寸信息
            vehicle_length = vehicle_state.car_footprint.oriented_box.length
            vehicle_width = vehicle_state.car_footprint.oriented_box.width

            # 获取车道ID
            lane_id = self._get_lane_id(vehicle_state.center, scenario)

        elif isinstance(vehicle_state, Agent):
            # 提取其他车辆速度信息
            velocity = vehicle_state.velocity

            # 提取其他车辆尺寸信息
            vehicle_length = vehicle_state.box.length
            vehicle_width = vehicle_state.box.width

            # 获取车道ID
            lane_id = self._get_lane_id(vehicle_state.box.center, scenario)

        else:
            raise ValueError(f"Unsupported vehicle state type: {type(vehicle_state)}")

        # 获取车头时距和车头间距信息
        preceding_vehicle_id, following_vehicle_id, space_headway, time_headway = self._get_headway_info(vehicle_state,
                                                                                                        lane_id)

        vehicle_data = {
            "Vehicle_ID": vehicle_id,
            "Frame_ID": iteration,
            "Total_Frames": total_iterations,
            "Global_Time": iteration * scenario.database_interval,
            "Local_X": vehicle_pose['x'],
            "Local_Y": vehicle_pose['y'],
            "Global_X": vehicle_pose['x'],
            "Global_Y": vehicle_pose['y'],
            "v_length": vehicle_length,
            "v_Width": vehicle_width,
            "v_Class": "ego" if isinstance(vehicle_state, EgoState) else vehicle_state.tracked_object_type.name.lower(),
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
            "Acceleration_x": vehicle_pose['acceleration_x'],
            "Acceleration_y": vehicle_pose['acceleration_y'],
            "Yaw_rate": vehicle_pose['angular_velocity'],
            "EPSG": vehicle_pose['epsg'],
        }

        return vehicle_data


    def _get_lane_id(self, pose, scenario):
        point = Point(pose.x, pose.y)

        # 获取地图数据
        map_data = scenario.map_api

        # 获取车道数据
        nearest_lane = map_data.get_one_map_object(point, SemanticMapLayer.LANE)

        # 检查是否找到车道
        if nearest_lane:
            lane = NuPlanLane(
                nearest_lane.id,
                map_data.get_available_map_objects()[0],
                map_data.get_available_map_objects()[SemanticMapLayer.LANE_CONNECTOR],
                map_data.get_available_map_objects()[SemanticMapLayer.BASELINE_PATHS],
                map_data.get_available_map_objects()[SemanticMapLayer.BOUNDARIES],
                map_data.get_available_map_objects()[SemanticMapLayer.STOP_LINE],
                map_data.get_available_map_objects()[SemanticMapLayer.LANE_CONNECTOR],
                map_data
            )

            # 检查点是否在车道的多边形内
            if lane.polygon.contains(point):
                return int(lane.index)

        return None

    def _get_headway_info(self, vehicle_state, lane_id):
        # You need to implement the logic to find the preceding and following vehicles,
        # and calculate the space and time headway based on the nuPlan API.
        # This will require additional information from the scenario and map API.
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


if __name__ == "__main__":
    # 设置数据集和地图的路径
    data_root = '/home/peter/GameFormer-Planner/nuplan/dataset/nuplan-v1.1'
    map_root = '/home/peter/GameFormer-Planner/nuplan/dataset/maps'

    # 设置数据库文件和地图版本
    db_files = [
        f"{data_root}/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db",
        f"{data_root}/splits/mini/2021.05.12.22.28.35_veh-35_00620_01164.db",
        # 添加更多数据库文件路径...
    ]
    map_version = "nuplan-maps-v1.0"  # 提供地图数据库版本
    sensor_root = None

    # 创建 NuPlanDBWrapper 对象
    nuplan_db_wrapper = NuPlanDBWrapper(data_root=data_root, map_root=map_root, db_files=db_files, map_version=map_version)

    # 创建 NuPlanScenarioBuilder 对象
    scenario_builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=map_root,
        db_files=db_files,
        sensor_root=sensor_root,
        map_version=map_version,
        scenario_mapping=ScenarioMapping(scenario_map={}, subsample_ratio_override=0.5),  # 0.5是子采样比例
    )

    # 创建 ScenarioFilter 对象
    scenario_filter = ScenarioFilter(*get_filter_parameters(1000, None, False))

    # 创建 WorkerPool 对象
    worker = SingleMachineParallelExecutor(use_process_pool=True)

    # 获取所有场景
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
    print(f"Number of scenarios: {len(scenarios)}")

    # 获取地图 API
    map_api = nuplan_db_wrapper.maps_db

    # 创建 DataProcessor 对象并处理数据
    processor = DataProcessor(scenarios, map_api)
    processor.process_and_save('nuplan_data.csv')