import os
import subprocess
import sys
import glob
import time
import gym
import random

import pygame
import cv2
from pygame.locals import *
from carla_env.generate_traffic import *
from carla_env.tools.hud import HUD
from carla_env.navigation.planner import RoadOption, compute_route_waypoints
from carla_env.navigation.controller import PIDLateralController, PIDLongitudinalController
from carla_env.wrappers import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from collections import deque
import itertools

intersection_routes = itertools.cycle(
    [(57, 81), (70, 11), (70, 12), (78, 68), (74, 41), (42, 73), (71, 62), (74, 40), (71, 77), (6, 12), (65, 52), (63, 80)])
eval_routes = itertools.cycle([(48, 21), (0, 72), (28, 83), (61, 39)])

discrete_actions = {
    0: [-1, 1], 1: [0, 1], 2: [1, 1], 3: [0, 0],
}

class CarlaRouteEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, host="127.0.0.1", port=2000,
                 viewer_res=(1120, 560), obs_res=(160, 80),
                 reward_fn=None,
                 observation_space=None,
                 encode_state_fn=None, decode_vae_fn=None,
                 fps=15, action_smoothing=0.0, action_space_type="continuous",
                 activate_spectator=True,
                 activate_lidar=True,
                 start_carla=True,
                 eval=False,
                 activate_render=True):

        self.carla_process = None
        if start_carla:
            if "CARLA_ROOT" not in os.environ:
                raise Exception("${CARLA_ROOT} has not been set!")
            carla_path = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
            launch_command = [carla_path]
            launch_command += ['-quality_level=Low']
            launch_command += ['-benchmark']
            launch_command += ["-fps=%i" % fps]
            launch_command += ['-RenderOffScreen']
            launch_command += ['-prefernvidia']
            launch_command += [f'-carla-world-port={port}']
            print("Running command:")
            print(" ".join(launch_command))
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print("Waiting for CARLA to initialize\n")

            # ./CarlaUE4.sh -quality_level=Low -benchmark -fps=15 -RenderOffScreen
            time.sleep(5)

        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res
        self.activate_render = activate_render

        # Setup gym environment
        self.action_space_type = action_space_type
        # HYBRID CONTROL: Decision-Based (STOP/GO)
        # 0: GO (Maintain Speed)
        # 1: STOP (Full Brake)
        self.action_space = gym.spaces.Discrete(2)

        self.observation_space = observation_space

        self.fps = fps
        self.action_smoothing = action_smoothing
        self.episode_idx = -2
        
        # LOGIC PARAMETERS
        self.ZONE_THRESHOLD = 30.0 # meters to traffic light
        self.decision = 0 # 0=GO, 1=STOP (For reward/logs)

        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.decode_vae_fn = None if not callable(decode_vae_fn) else decode_vae_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.max_distance = 3000  # m
        self.activate_spectator = True
        self.activate_lidar = activate_lidar
        self.prev_object_distances = {}
        self.eval = eval
        self.world = None

        # Traffic Light State
        self.tl_state = 2 # Default Green
        # Traffic Light State
        self.tl_state = 2 # Default Green
        self.tl_dist = 100.0

        # Pedestrian State
        self.walker = None # Deprecated, keeping for safety or single ref
        self.walkers_list = []
        self.walker_controllers_list = []
        self.pedestrian_ahead = False
        self.ped_dist = 100.0 # Internal only
        self.ped_recency_timer = 0 # Cooldown timer
        self.PED_COOLDOWN = 150 # Steps between interactions


        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(60.0)

            # Create world wrapper
            self.world = World(self.client)

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1 / self.fps
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.client.reload_world(False)  # reload map keeping the world settings

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0],
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))
            
            # --- GLOBAL PEDESTRIAN SPAWNING ---
            self._spawn_global_pedestrians(num_walkers=40)

            # Initialize PID Contollers for Rule-Based Navigation
            self.lat_controller = PIDLateralController(self.vehicle.actor, K_P=1.95, K_D=0.2, K_I=0.07, dt=1.0/self.fps)
            self.lon_controller = PIDLongitudinalController(self.vehicle.actor, K_P=1.0, K_D=0.0, K_I=0.0, dt=1.0/self.fps)

            #self.traffic_manager = self.client.get_trafficmanager(8000)  # Remove traffic manager

            # Generate traffic before starting the environment
            #self.vehicles, self.walkers = generate_traffic(self.client, self.world, self.traffic_manager, num_vehicles=50, num_walkers=20) # Remove traffic generation

            # Create hud and initialize pygame for visualization
            if self.activate_render:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                pygame.event.pump()
                pygame.display.flip()

                self.clock = pygame.time.Clock()
                self.hud = HUD(width, height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)

            seg_settings = {}
            if "seg_camera" in self.observation_space.keys():
                seg_settings.update({
                    'camera_type': "sensor.camera.semantic_segmentation",
                    'custom_palette': True
                })
            self.dashcam = Camera(self.world, out_width, out_height,
                                  transform=sensor_transforms["dashboard"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image(e),
                                  **seg_settings)

            if self.activate_spectator:
                self.camera = Camera(self.world, width, height,
                                     transform=sensor_transforms["spectator"],
                                     attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e))
            if self.activate_lidar:
                self.lidar = Lidar(self.world, transform=sensor_transforms["lidar"],
                                   attach_to=self.vehicle, on_recv_image=lambda e: self._set_lidar_data(e))

            self.obstacle_detector = ObstacleDetector(
                world=self.world,
                vehicle=self.vehicle,
                distance=15.0,  # Detection range (adjust as needed)
                hit_radius=1.0,  # Sensor width (adjust for sensitivity)
                on_detect=lambda ttc, obst_type: self.on_obstacle_detected(ttc, obst_type)  # Callback function
            )

        except Exception as e:
            self.close()
            raise e
        # Reset env to set initial state
        self.reset()

    def reset(self, seed=None, options=None, is_training=False):
        # Create new route
        # Reset state flags BEFORE generating new route
        self.terminal_state = False
        self.success_state = False
        self.collision_detected = False
        self.closed = False
        self.extra_info = []
        self.observation = self.observation_buffer = None
        self.viewer_image = self.viewer_image_buffer = None
        self.lidar_data = self.lidar_data_buffer = None
        self.step_count = 0
        self.step_count = 0
        self.last_ttc = 1000.0 # Reset safety trigger

        # Reset Pedestrian State - destroy old if exists
        # self._destroy_walker() # Don't destroy global walkers on reset
        self.pedestrian_ahead = False
        self.ped_dist = 100.0
        # self.ped_recency_timer = 0


        # Create new route (teleport and physics reset happens here)
        self.num_routes_completed = -1
        self.episode_idx += 1
        self.new_route()
        self.world.tick()

        # Init metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0

        #reward parameters
        self.prev_speed = 0
        self.delta_v_accum = 0
        self.last_ttc = 1000.0
        self.obst_type = ""
        self.prev_distance_to_goal = self.distance_to_goal
        #self.traffic_light_state = carla.TrafficLightState.Unknown  # Remove traffic light state
        #self.following_distance = float('inf')  # Remove following distance
        #self.other_vehicle_speed = 0.0  # Remove other vehicle speed

        # Return initial observation
        time.sleep(0.2)
        obs = self.step(None)[0]
        time.sleep(0.2)
        info = {}  # Add any useful info if needed
        return obs, info

    def new_route(self):
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        self.vehicle.control.brake = float(0.0) # Ensure brake is released
        self.vehicle.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0)) # Force apply
        self.vehicle.set_simulate_physics(False)  # Reset the car's physics

        # Clear sensor history
        if hasattr(self.vehicle, 'collision_sensor'):
            self.vehicle.collision_sensor.history = []


        # Generate waypoints along the lap
        if not self.eval:
            if self.episode_idx % 2 == 0 and self.num_routes_completed == -1:
                spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(intersection_routes)]
            else:
                spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)
        else:
            spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(eval_routes)]
        route_length = 1
        while route_length <= 1:
            self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in
                                          spawn_points_list]
            start_location = self.start_wp.transform.location
            end_location = self.end_wp.transform.location
            self.distance_to_goal = ((end_location.x - start_location.x)**2 + (end_location.y - start_location.y)**2 + (end_location.z - start_location.z)**2)
            self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
            route_length = len(self.route_waypoints)
            if route_length <= 1:
                spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)

        self.distance_from_center_history = deque(maxlen=30)

        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_transform(self.start_wp.transform)
        time.sleep(0.2)
        self.vehicle.set_simulate_physics(True)

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        
        # Cleanup walkers
        self._destroy_global_walkers()
        
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):
        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation

        # Tick render clock
        self.clock.tick()
        self.hud.tick(self.world, self.clock)

        # Get maneuver name
        if self.current_road_maneuver == RoadOption.LANEFOLLOW:
            maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:
            maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:
            maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT:
            maneuver = "Straight"
        else:
            maneuver = "INVALID"

        # Add metrics to HUD
        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Maneuver:        % 11s" % maneuver,
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
            "Total reward:        % 7.2f" % self.total_reward,
        ])
        if self.activate_spectator:
            # Blit image from spectator camera
            self.viewer_image = self._draw_path(self.camera, self.viewer_image)
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))
            # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation.shape[:2]
        pos_observation = (self.display.get_size()[0] - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos_observation)

        pos_vae_decoded = (self.display.get_size()[0] - 2 * obs_w - 10, 10)
        if self.decode_vae_fn:
            self.display.blit(pygame.surfarray.make_surface(self.observation_decoded.swapaxes(0, 1)), pos_vae_decoded)

        if self.activate_lidar:
            lidar_h, lidar_w = self.lidar_data.shape[:2]
            pos_lidar = (self.display.get_size()[0] - obs_w - 10, 100)
            self.display.blit(pygame.surfarray.make_surface(self.lidar_data.swapaxes(0, 1)), pos_lidar)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list

        # Render to screen
        pygame.display.flip()

    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        
        # Update Pedestrian Logic (Spawn/Despawn)
        self._update_pedestrian_logic()

        # Update Pedestrian Info (Detection) for this frame
        self._update_pedestrian_info()

        # Take action
        self.last_ttc = 1000.0 # Reset safety state for this frame

        if action is not None:
            # Create new route on route completion
            if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                if not self.eval:
                    self.new_route()
                else:
                    self.success_state = True

            # --- HYBRID CONTROL LOGIC ---
            # 1. Rule-Based Steering (PID)
            if self.current_waypoint_index < len(self.route_waypoints) - 1:
                 lookahead_idx = min(self.current_waypoint_index + 2, len(self.route_waypoints) - 1)
                 target_wp = self.route_waypoints[lookahead_idx][0]
            else:
                 target_wp = self.current_waypoint
            steer = self.lat_controller.run_step(target_wp)

            # 2. Decision Logic (STOP/GO)
            # Update TL info first to use in decision
            self._update_traffic_light_info() 
            
            # Check if inside Decision Zone
            # We use self.tl_dist (updated below usually, but we need it here. 
            # Ideally _update_traffic_light_info() should be called start of step or end of previous. 
            # It was called at end of step. Let's assume it's valid from previous step. 
            # But for step 0 it might be 100.
            
            in_zone = (self.tl_dist < self.ZONE_THRESHOLD) and (self.tl_state != 2) # Only care if NOT Green? Or always?
            # Requirement: "When the vehicle is NOT near any traffic light: FORCE decision = GO"
            # It implies distance check only.
            in_zone = (self.tl_dist < self.ZONE_THRESHOLD)
            
            # --- PEDESTRIAN OVERRIDE ---
            if self.pedestrian_ahead:
                in_zone = True # Force RL control
            
            final_action = 0 # Default GO

            
            final_action = 0 # Default GO
            
            if not in_zone:
                # FORCE GO
                final_action = 0
            else:
                # RL DECISION
                final_action = action
                
            self.decision = final_action

            # 3. Execution
            throttle = 0.0
            brake = 0.0
            target_speed_val = 25.0 # km/h
            
            if final_action == 0: # GO
                throttle = self.lon_controller.run_step(target_speed_val)
                brake = 0.0
            elif final_action == 1: # STOP
                throttle = 0.0
                brake = 1.0 # Smooth brake handled below

            # 4. Safety Override (Collision)
            if hasattr(self, 'last_ttc') and self.last_ttc < 1.5:
                throttle = 0.0
                brake = 1.0
                self.decision = 1 # Update decision to reflect reality (or keep RL decision? Keep RL decision for reward)

            # Apply smoothing
            self.vehicle.control.steer = float(steer)
            
            if float(throttle) == 0.0:
                self.vehicle.control.throttle = 0.0
            else:
                 self.vehicle.control.throttle = smooth_action(self.vehicle.control.throttle, float(throttle), self.action_smoothing)

            if float(brake) == 0.0:
                self.vehicle.control.brake = 0.0
            else:
                self.vehicle.control.brake = smooth_action(self.vehicle.control.brake, float(brake), self.action_smoothing)

        # Tick game
        self.world.tick()

        # Get most recent observation and viewer image
        self.observation = self._get_observation()
        if self.activate_spectator:
            self.viewer_image = self._get_viewer_image()

        if self.activate_lidar:
            self.lidar_data = self._get_lidar_data()

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of closest waypoint on the route
        self.prev_waypoint_index = self.current_waypoint_index
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:  # Did we pass the waypoint?
                waypoint_index += 1  # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index

        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)

        # Calculate deviation from center of the lane
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        # Calculate distance traveled
        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()
        self.prev_distance_to_goal = self.distance_to_goal

        #Delta speed
        current_speed = self.vehicle.get_speed()
        self.delta_v_accum += self.prev_speed - current_speed
        #print(f"delta_v_accum:{self.delta_v_accum}")
        self.prev_speed = current_speed

        # Terminal on max distance
        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True

        self.distance_from_center_history.append(self.distance_from_center)

        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        #print(f"episode: {self.episode_idx} last reward: {self.last_reward}")
        if np.isinf(self.last_reward) and self.last_reward < 0:
            self.total_reward = 0
        else:
            self.total_reward += self.last_reward
        #print(f"total reward: {self.total_reward}")

        # Encode the state
        encoded_state = self.encode_state_fn(self)
        if self.decode_vae_fn:
            self.observation_decoded = self.decode_vae_fn(encoded_state['vae_latent'])
        self.step_count += 1

        # DEBUG: Draw path
        # self._draw_path_server(life_time=1.0, skip=8)
        # DEBUG: Draw current waypoint
        # self.world.debug.draw_point(self.current_waypoint.transform.location + carla.Location(z=1.25), size=0.1,color=carla.Color(0, 255, 255), life_time=2.0, persistent_lines=False)

        # Check for ESC press
        if self.activate_render:
            pygame.event.pump()
            if pygame.key.get_pressed()[K_ESCAPE]:
                self.close()
                self.terminal_state = True
            self.render()

        # Update Traffic Light Info for Next State (MOVED UP IF NEEDED but okay here for observation)
        # self._update_traffic_light_info() # Already called inside step logic for decision

        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count),
            'tl_state': self.tl_state,
            'tl_dist': self.tl_dist,
            'tl_dist': self.tl_dist,
            'decision': self.decision,
            'pedestrian_ahead': self.pedestrian_ahead
        }
        return encoded_state, self.last_reward, self.terminal_state or self.success_state, False, info

    def _draw_path_server(self, life_time=60.0, skip=0):

        for i in range(0, len(self.route_waypoints) - 1, skip + 1):
            z = 30.25
            w0 = self.route_waypoints[i][0]
            w1 = self.route_waypoints[i + 1][0]
            self.world.debug.draw_line(
                w0.transform.location + carla.Location(z=z),
                w1.transform.location + carla.Location(z=z),
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=False)
            self.world.debug.draw_point(
                w0.transform.location + carla.Location(z=z), 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        self.world.debug.draw_point(
            self.route_waypoints[-1][0].transform.location + carla.Location(z=z), 0.1,
            carla.Color(0, 0, 255),
            life_time, False)

    def _draw_path(self, camera, image):

        vehicle_vector = vector(self.vehicle.get_transform().location)
        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = int(camera.actor.attributes['image_size_x'])
        image_h = int(camera.actor.attributes['image_size_y'])
        fov = float(camera.actor.attributes['fov'])
        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
            waypoint_vector = vector(waypoint_location)
            if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
                continue
            # Calculate the camera projection matrix to project from 3D -> 2D
            K = build_projection_matrix(image_w, image_h, fov)
            x, y = get_image_point(waypoint_location, K, world_2_camera)
            #print(f"get_image_point() output: {x},{y} ")
            #print(f"get_image_point() raw type: {type(x)}, {type(y)}")
            if i == len(self.route_waypoints) - 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            x = int(x)
            y = int(y)
            image = cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
        return image

    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _get_lidar_data(self):
        while self.lidar_data_buffer is None:
            pass
        image = self.lidar_data_buffer.copy()
        self.lidar_data_buffer = None
        return image

    def _on_collision(self, event):
        if get_actor_display_name(event.other_actor) != "Road":
            self.terminal_state = True
            self.collision_detected = True
        if self.activate_render:
            self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        if self.activate_render:
            self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _set_lidar_data(self, image):
        self.lidar_data_buffer = image

    def attempt_overtake(self):
        """
        Checks if overtaking is possible and executes it.
        """
        # Get current vehicle location and waypoints
        current_location = self.vehicle.get_transform().location
        current_wp = self.world.map.get_waypoint(current_location)

        # Check for a left or right lane
        left_wp = current_wp.get_left_lane()
        right_wp = current_wp.get_right_lane()

        # Try to change lane if it's free
        if left_wp and left_wp.lane_type == carla.LaneType.Driving:
            #print("Overtaking using left lane...")
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.3))  # Steer left
            time.sleep(2)  # Wait for lane change
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=0.0))  # Straighten vehicle

        elif right_wp and right_wp.lane_type == carla.LaneType.Driving:
            #print("Overtaking using right lane...")
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.3))  # Steer right
            time.sleep(2)
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=0.0))

        else:
            x = 1
            #print("No available lane to overtake. Maintaining speed.")

    def get_following_distance(self):
        """
        Approximates the distance to the vehicle directly in front.
        This is a simplified calculation and might need refinement.
        """
        ego_location = self.vehicle.get_transform().location
        ego_speed_kmh = self.get_speed(self.vehicle.actor)  # Corrected: Use get_speed()
        min_distance = float('inf')  # Initialize with a very large number
        self.other_vehicle_speed = 0.0  # Initialize

        for actor in self.world.world.get_actors().filter('*vehicle*'):  # Iterate through all vehicles
            if actor.id == self.vehicle.actor.id:  # Skip the ego vehicle itself
                continue

            other_vehicle_location = actor.get_transform().location
            other_vehicle_speed_kmh = self.get_speed(actor)  # Corrected: Use get_speed()

            # 1. Check if the other vehicle is ahead (simplified)
            ego_forward_vector = self.vehicle.get_transform().get_forward_vector()
            direction_to_other = other_vehicle_location - ego_location
            dot_product = ego_forward_vector.x * direction_to_other.x + ego_forward_vector.y * direction_to_other.y

            if dot_product > 0:  # Other vehicle is in front (approximately)
                distance = ego_location.distance(other_vehicle_location)
                if distance < min_distance:
                    min_distance = distance
                    self.other_vehicle_speed = other_vehicle_speed_kmh

        if min_distance == float('inf'):
            return 1000  # Return a large number if no vehicle is ahead
        return min_distance

    def get_speed(self, vehicle):
        """
        Calculates the speed of a CARLA vehicle in km/h.
        """
        velocity = vehicle.get_velocity()
        speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        return speed_kmh

    def get_traffic_light_state(self):
        """
        Gets the state of the closest traffic light affecting the vehicle.
        """
        traffic_light = self.vehicle.get_traffic_light()
        if traffic_light is not None:
            return traffic_light.get_state()
        return carla.TrafficLightState.Unknown

    def on_obstacle_detected(self, ttc, obst_type):

        # Store TTC for use in the environment
        self.last_ttc = ttc  # Store latest TTC value
        self.obst_type = obst_type
        if obst_type == "vehicle":
            if ttc < 3.0:  # If time to collision is less than 3 seconds
                #print("Vehicle ahead detected! Slowing down...")
                self.vehicle.apply_control(carla.VehicleControl(throttle=(self.vehicle.control.throttle/2), steer=0.0))  # Reduce speed

                # Try to change lane
                # self.attempt_overtake() # Disable overtake in hybrid mode
                pass

    def _update_traffic_light_info(self):
        """
        Updates self.tl_state and self.tl_dist
        State mapping: 0=Red, 1=Yellow, 2=Green
        """
        actor = self.vehicle.actor
        self.tl_dist = 100.0 # Default far
        self.tl_state = 2 # Default Green

        # Find nearest traffic light
        # This is a simplified check. CARLA vehicles have built-in traffic light detection.
        # But we need distance.
        if actor.is_at_traffic_light():
            light = actor.get_traffic_light()
            if light:
                state = light.get_state()
                # CARLA states: Red, Yellow, Green, Off, Unknown
                if state == carla.TrafficLightState.Red:
                    self.tl_state = 0
                elif state == carla.TrafficLightState.Yellow:
                    self.tl_state = 1
                elif state == carla.TrafficLightState.Green:
                    self.tl_state = 2
                else:
                     self.tl_state = 2 # Treat others as green

                # Distance? is_at_traffic_light() means we are in the trigger box.
                # Usually that's close (e.g. < 5-10m).
                # To get precise distance we need landmarks, but simply setting a small distance if "at" light works for now.
                # However, RL needs to see it Coming.
                # Better approach: Iterate all traffic lights and find closest one ahead.
                
                # For now, relying on 'is_at_traffic_light' might be too late for braking. 
                # Converting to the requested logic:
                
                waypoints = self.world.map.get_waypoint(actor.get_location())
                # We can use the vehicle's traffic light state directly if available
                
                # Use simplified "distance" based on trigger volume
                # If we are in trigger volume, distance is low.
                
                # IMPROVEMENT: Use the limit trigger volume
                pass

        # Better simple approach for now if no complex query:
        # Use valid traffic light from carla
        last_tl = actor.get_traffic_light()
        if last_tl:
             # Calculate distance
             tl_loc = last_tl.get_transform().location
             veh_loc = actor.get_location()
             dist = veh_loc.distance(tl_loc)
             self.tl_dist = dist
             
             st = last_tl.get_state()
             if st == carla.TrafficLightState.Red:
                 self.tl_state = 0
             elif st == carla.TrafficLightState.Yellow:
                 # Yellow Logic: If close -> Red, else max Green
                 if dist < 10.0: # Close to stop line approx
                     self.tl_state = 0 # Treat as Red
                 else:
                     self.tl_state = 2 # Treat as Green (risk it)
             elif st == carla.TrafficLightState.Green:
                 self.tl_state = 2
             else:
                 self.tl_state = 2
        else:
             self.tl_state = 2 # No light = Green
             self.tl_dist = 50.0 # Far

             self.tl_state = 2 # No light = Green
             self.tl_dist = 50.0 # Far

             self.tl_dist = 50.0 # Far

    def _destroy_global_walkers(self):
        for controller in self.walker_controllers_list:
            if controller:
                controller.stop()
                controller.destroy()
        for walker in self.walkers_list:
            if walker:
                walker.destroy()
        self.walker_controllers_list = []
        self.walkers_list = []
        print("Destroyed global walkers.")

    def _spawn_global_pedestrians(self, num_walkers=150):
        """
        Spawns pedestrians globally on the map.
        """
        try:
            self._destroy_global_walkers()
            
            # Detect Map
            map_name = self.world.map.name
            print(f"Detected Map: {map_name}")
            
            blueprints = self.world.world.get_blueprint_library().filter("walker.pedestrian.*")
            controller_bp = self.world.world.get_blueprint_library().find('controller.ai.walker')
            
            spawn_points = []
            
            # Try to get random locations for navigation (including road areas)
            for i in range(num_walkers):
                loc = self.world.world.get_random_location_from_navigation()
                if loc:
                    spawn_points.append(carla.Transform(loc))
                else:
                    # If no navigation point, try spawning on road edges for more random behavior
                    waypoint = self.world.map.get_waypoint(self.vehicle.get_transform().location)
                    if waypoint:
                        random_waypoint = waypoint.next(random.uniform(10, 50))[0]
                        if random_waypoint:
                            spawn_points.append(carla.Transform(random_waypoint.transform.location + carla.Location(z=0.5)))
            
            # Batch spawn actors
            batch = []
            for sp in spawn_points:
                walker_bp = random.choice(blueprints)
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                batch.append(carla.command.SpawnActor(walker_bp, sp))
            
            results = self.client.apply_batch_sync(batch, True)
            
            for i, res in enumerate(results):
                if not res.error:
                    self.walkers_list.append(self.world.world.get_actor(res.actor_id))
            
            print(f"Spawned {len(self.walkers_list)} walkers.")
            
            # Spawn controllers
            batch = []
            for walker in self.walkers_list:
                batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), walker))
            
            results = self.client.apply_batch_sync(batch, True)
            
            for i, res in enumerate(results):
                if not res.error:
                    controller = self.world.world.get_actor(res.actor_id)
                    self.walker_controllers_list.append(controller)
                    
            print(f"Spawned {len(self.walker_controllers_list)} walker controllers.")
            
            # Start controllers with aggressive crossing behavior
            for controller in self.walker_controllers_list:
                controller.start()
                # Initial destination - force road crossing
                dest = self.world.world.get_random_location_from_navigation()
                if not dest:
                    # Force crossing near vehicle if no navigation point
                    vehicle_loc = self.vehicle.get_transform().location
                    dest = vehicle_loc + carla.Location(x=random.uniform(-20, 20), y=random.uniform(-20, 20))
                if dest:
                    controller.go_to_location(dest)
                    # Higher random speeds for more unpredictable behavior
                    controller.set_max_speed(1.5 + random.random() * 1.5)  # 1.5-3.0 m/s

        except Exception as e:
            print(f"Error spawning global pedestrians: {e}")

    def _update_pedestrian_logic(self):
        """
        Manages Global Walkers (Destinations) - Forces sudden crossings away from traffic lights.
        """
        # Aggressively update destinations to force constant road crossing
        if random.random() < 0.25: # 25% chance per frame to update a walker
            if self.walker_controllers_list:
                # Update multiple walkers at once for chaos
                num_updates = random.randint(1, min(8, len(self.walker_controllers_list)))
                for _ in range(num_updates):
                    idx = random.randint(0, len(self.walker_controllers_list)-1)
                    controller = self.walker_controllers_list[idx]
                    walker = self.walkers_list[idx]
                    
                    # Get current walker position
                    walker_loc = walker.get_location()
                    vehicle_loc = self.vehicle.get_transform().location
                    
                    # Find a road section AWAY from traffic lights for sudden crossing
                    dest = self._find_non_intersection_crossing_point(walker_loc, vehicle_loc)
                    
                    if dest:
                        controller.go_to_location(dest)
                        # Very high speeds for sudden dashes
                        controller.set_max_speed(2.5 + random.random() * 2.0)  # 2.5-4.5 m/s (running speed)
    
    def _find_non_intersection_crossing_point(self, walker_loc, vehicle_loc):
        """
        Find a crossing point away from traffic lights and intersections.
        """
        try:
            # Get current waypoint
            current_wp = self.world.map.get_waypoint(walker_loc)
            if not current_wp:
                return None
            
            # Check if near intersection/traffic light
            is_near_intersection = current_wp.is_junction
            
            # Find a road section that's NOT at an intersection
            for _ in range(20):  # Try 20 times to find a good spot
                # Random distance from vehicle (but not too far)
                distance = random.uniform(15, 60)
                
                # Get random waypoint at that distance
                vehicle_wp = self.world.map.get_waypoint(vehicle_loc)
                if vehicle_wp:
                    next_wps = vehicle_wp.next(distance)
                    if next_wps:
                        target_wp = random.choice(next_wps)
                        
                        # Make sure it's not at an intersection
                        if not target_wp.is_junction:
                            # Add some perpendicular offset for crossing behavior
                            perpendicular = target_wp.transform.get_right_vector()
                            offset_dist = random.uniform(-10, 10)
                            dest = target_wp.transform.location + perpendicular * offset_dist
                            
                            # Add some randomness to Z for realism
                            dest.z += random.uniform(-0.5, 0.5)
                            
                            return dest
            
            # Fallback: random location but avoid obvious intersection areas
            dest = vehicle_loc + carla.Location(
                x=random.uniform(-40, 40), 
                y=random.uniform(-40, 40),
                z=random.uniform(-1, 1)
            )
            return dest
            
        except Exception as e:
            print(f"Error finding crossing point: {e}")
            return None

    def _update_pedestrian_info(self):
        """
        Updates self.pedestrian_ahead and self.ped_dist scanning ALL walkers.
        """
        self.pedestrian_ahead = False
        self.ped_dist = 100.0
        
        if not self.walkers_list:
            return
            
        v_loc = self.vehicle.get_transform().location
        v_fwd = self.vehicle.get_transform().get_forward_vector()
        
        min_dist = 100.0
        detected = False
        
        for walker in self.walkers_list:
            # Optimization: First check euclidian distance
            w_loc = walker.get_location()
            vec_to_w = w_loc - v_loc
            dist = vec_to_w.length()
            
            if dist > 60.0:
                continue
                
            # Check angle (must be in front)
            dot = v_fwd.x * vec_to_w.x + v_fwd.y * vec_to_w.y
            if dot < 0:
                 continue
                 
            # Lateral Check
            cross = v_fwd.x * vec_to_w.y - v_fwd.y * vec_to_w.x
            lat_dist = abs(cross)
            
            if lat_dist < 2.5: # In lane corridor
                if dist < min_dist:
                    min_dist = dist
                    detected = True
        
        if detected:
            self.pedestrian_ahead = True
            self.ped_dist = min_dist



