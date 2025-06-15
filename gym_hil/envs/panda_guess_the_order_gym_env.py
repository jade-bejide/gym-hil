#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gymnasium as gym
from typing import Any, Dict, Literal, Tuple
from pathlib import Path
import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec
import time

_PANDA_HOME = np.asarray((0, 0.195, 0, -2.43, 0, 2.62, 0.785))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])

class PandaGuessTheOrderGymEnv(FrankaGymEnv):
    """Environment for a Panda robot picking up a cube."""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_block_position: bool = False,
    ):
        self.reward_type = reward_type

        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
            xml_path= Path(__file__).parent.parent / "assets" / "scene2.xml"
        )

        # Task-specific setup
        self._block_z = self._model.geom("block1").size[2]
        self._random_block_position = random_block_position

        # Setup observation space properly to match what _compute_observation returns
        # Observation space design:
        #   - "state":  agent (robot) configuration as a single Box
        #   - "environment_state": block position in the world as a single Box
        #   - "pixels": (optional) dict of camera views if image observations are enabled

        agent_dim = self.get_robot_state().shape[0]
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                0,
                                255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                0,
                                255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                    "agent_pos": agent_box,
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_box,
                    "environment_state": env_box,
                }
            )

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        # Ensure gymnasium internal RNG is initialized when a seed is provided
        super().reset(seed=seed)

        mujoco.mj_resetData(self._model, self._data)

        # Reset the robot to home position
        self.reset_robot()
        NO_BLOCKS=5

        positions_coords = np.linspace(-0.7, 0.7, NO_BLOCKS)
        np.random.shuffle(positions_coords)

        central_block = np.random.uniform(*_SAMPLING_BOUNDS)
        # Sample a new block position
        if self._random_block_position:
            # self._data.jnt("block1").qpos[:3] = (*block_xy, self._block_z)
            blocks = [f"block{i}" for i in range(1,NO_BLOCKS+1)]
            np.random.shuffle(blocks)
            # Add in the target positions
            targets = [f"target{i}" for i in range(1,NO_BLOCKS+1)]
            np.random.shuffle(targets)

            for block, target, pos in zip(blocks, targets, positions_coords):
                block_coords = np.array([central_block[0], central_block[1]+pos])
                target_coords = np.array([central_block[0]+0.25, central_block[1]+pos])
                self._data.joint(block).qpos[:3] = (*block_coords, self._block_z)
                self._data.joint(target).qpos[:3] = (*target_coords, self._block_z)
            
        else:
            # Not applicable for PandaGuessTheOrder
            pass
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height
        self._z_init = self._data.sensor("block1_pos").data[2]
        self._z_success = self._z_init + 0.1

        obs = self._compute_observation()

        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Apply the action to the robot
        self.apply_action(action)

        # Compute observation, reward and termination
        obs = self._compute_observation()
        rew = self._compute_reward2()
        success = self._is_success2()

        if self.reward_type == "sparse":
            success = rew == 1.0

        # Check if block is outside bounds
        block_pos = self._data.sensor("block1_pos").data
        exceeded_bounds = np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.05)) or np.any(
            block_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.05)
        )

        terminated = bool(success or exceeded_bounds)

        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        """Compute the current observation."""
        # Create the dictionary structure that matches our observation space
        observation = {}

        # Get robot state
        robot_state = self.get_robot_state().astype(np.float32)

        # Assemble observation respecting the newly defined observation_space
        block_pos = self._data.sensor("block1_pos").data.astype(np.float32)

        if self.image_obs:
            # Image observations
            front_view, wrist_view = self.render()
            observation = {
                "pixels": {"front": front_view, "wrist": wrist_view},
                "agent_pos": robot_state,
            }
        else:
            # State-only observations
            observation = {
                "agent_pos": robot_state,
                "environment_state": block_pos,
            }

        return observation

    def _compute_reward(self) -> float:
        """Compute reward based on current state."""
        block_pos = self._data.sensor("block1_pos").data

        if self.reward_type == "dense":
            tcp_pos = self._data.sensor("2f85/pinch_pos").data
            dist = np.linalg.norm(block_pos - tcp_pos)
            r_close = np.exp(-20 * dist)
            r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
            r_lift = np.clip(r_lift, 0.0, 1.0)
            return 0.3 * r_close + 0.7 * r_lift
        else:
            lift = block_pos[2] - self._z_init
            return float(lift > 0.1)
        


    def _is_success(self) -> bool:
        """Check if the task is successfully completed."""
        block_pos = self._data.sensor("block1_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        lift = block_pos[2] - self._z_init
        return dist < 0.05 and lift > 0.1
    
    # second reward function and terminating condition based on target position

    def get_sensors(self):
        block_sensors = [self._data.sensor(f"block{i}_pos") for i in range(1,6)]
        target_sensors = [self._data.sensor(f"target{i}_pos") for i in range(1,6)]
        return block_sensors, target_sensors
    
    def _compute_reward2(self) -> float:
        block_sensors, target_sensors = self.get_sensors()
        pairs = zip(block_sensors, target_sensors)
        distances = list(map(lambda pair: np.exp(-20 * np.linalg.norm(pair[0].data-pair[1].data)), pairs))
        return sum(distances)
    
    def _is_success2(self) -> bool:
        block_sensors, target_sensors = self.get_sensors()
        pairs = zip(block_sensors, target_sensors)
        distances = list(map(lambda pair: np.linalg.norm(pair[0].data-pair[1].data), pairs))

        return all(list(map(lambda dist: dist < 0.05, distances)))




# Enables keyboard control of Gym environment - for episode recording
def human_in_the_loop():
    env = PandaGuessTheOrderGymEnv(render_mode="human", random_block_position=True, image_obs=True)
    obs, _ = env.reset()

    print("Observation keys:", list(obs.keys()))
    if "pixels" in obs:
        print("Pixels keys:", list(obs["pixels"].keys()))

    env_id = "gym_hil/PandaGuessTheOrderKeyboard-v0"
    env = gym.make(
        env_id,
        render_mode="human",
        image_obs=True,
        use_gamepad=False,
        max_episode_steps=50000
    )

    # obs, _ = env.reset()
    dummy_action = np.zeros(4, dtype=np.float32)
    # This ensures the "stay gripper" action is set when the intervention button is not pressed
    dummy_action[-1] = 1

    try:
        while True:
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(dummy_action)

            # Print some feedback
            if info.get("succeed", False):
                print("\nSuccess! Block has been picked up.")

            # If auto-reset is disabled, manually reset when episode ends
            # if terminated or truncated:
            #     print("Episode ended, resetting environment")
            #     obs, _ = env.reset()

            # Add a small delay to control update rate
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()
        print("Session ended")


if __name__ == "__main__":
    from gym_hil import PassiveViewerWrapper
    from gym_hil.wrappers.hil_wrappers import InputsControlViewerWrapper
    # human_in_the_loop()
    env = PandaGuessTheOrderGymEnv(render_mode="human", random_block_position=True)
    env = InputsControlViewerWrapper(env, use_gamepad=False)
    env.reset()
    for _ in range(1000):
        env.step(np.random.uniform(-1, 1, 7))
    env.close()
