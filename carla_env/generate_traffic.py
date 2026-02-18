#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
from numpy import random

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def generate_traffic(client, world, traffic_manager, num_vehicles=30, num_walkers=10, asynch=False):
    """
    Function to generate traffic in CARLA.

    Args:
        client (carla.Client): The CARLA client.
        world (carla.World): The CARLA world object.
        traffic_manager (carla.TrafficManager): The CARLA Traffic Manager.
        num_vehicles (int): Number of vehicles to spawn.
        num_walkers (int): Number of pedestrians to spawn.
        asynch (bool): Whether to run in asynchronous mode.
    """

    vehicles_list = []
    walkers_list = []
    all_id = []

    try:
        settings = world.get_settings()
        synchronous_master = not asynch

        if not asynch:
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05

        world.apply_settings(settings)

        # Spawn vehicles
        blueprints = world.get_blueprint_library().filter('vehicle.*')
        spawn_points = world.get_map().get_spawn_points()
        num_spawn = min(num_vehicles, len(spawn_points))

        batch = []
        for i in range(num_spawn):
            blueprint = random.choice(blueprints)
            transform = spawn_points[i]
            batch.append(carla.command.SpawnActor(blueprint, transform)
                         .then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port())))

        responses = client.apply_batch_sync(batch, synchronous_master)
        vehicles_list = [response.actor_id for response in responses if not response.error]

        print(f"Spawned {len(vehicles_list)} vehicles.")

        # Spawn Walkers
        walker_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
        walker_batch = []
        for _ in range(num_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc:
                spawn_point.location = loc
                walker_blueprints = random.choice(walker_blueprints)
                walker_batch.append(carla.command.SpawnActor(walker_blueprints, spawn_point))

        responses = client.apply_batch_sync(walker_batch, synchronous_master)
        walkers_list = [response.actor_id for response in responses if not response.error]

        print(f"Spawned {len(walkers_list)} pedestrians.")

        return vehicles_list, walkers_list

    except Exception as e:
        print(f"Error in traffic generation: {e}")

    return [], []

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
