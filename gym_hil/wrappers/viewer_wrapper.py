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

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
import mujoco.viewer

import sys


class PassiveViewerWrapper(gym.Wrapper):
    """Gym wrapper that opens a passive MuJoCo viewer automatically.

    The wrapper starts a MuJoCo viewer in passive mode as soon as the
    environment is created so the user no longer needs to use
    ``mujoco.viewer.launch_passive`` or any context–manager boiler-plate.

    The viewer is kept in sync after every ``reset`` and ``step`` call and is
    closed automatically when the environment itself is closed or deleted.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        show_left_ui: bool = False,
        show_right_ui: bool = False,
    ) -> None:
        super().__init__(env)

        # Launch the interactive viewer.  We expose *model* and *data* from the
        # *unwrapped* environment to make sure we operate on the base MuJoCo
        # objects even if other wrappers have been applied before this one.
        self._viewer = mujoco.viewer.launch_passive(
            env.unwrapped.model,
            env.unwrapped.data,
            # show_left_ui=show_left_ui,
            # show_right_ui=show_right_ui,
        )

        # Make sure the first frame is rendered.
        self._viewer.sync()

    # ---------------------------------------------------------------------
    # Gym API overrides

    def reset(self, **kwargs):  # type: ignore[override]
        observation, info = self.env.reset(**kwargs)
        self._viewer.sync()
        return observation, info

    def step(self, action):  # type: ignore[override]
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._viewer.sync()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:  # type: ignore[override]
        """Close both the passive viewer and the underlying gym environment.

        MuJoCo's `Renderer` gained a `close()` method only in recent versions
        (>= 2.3.0).  When running with an older MuJoCo build the renderer
        instance stored inside `env.unwrapped._viewer` does not provide this
        method which causes `AttributeError` when the environment is closed.

        To remain version-agnostic we:
          1. Manually dispose of the underlying viewer *only* if it exposes a
             `close` method.
          2. Remove the reference from the environment so that a subsequent
             call to `env.close()` will not fail.
          3. Close our own passive viewer handle.
          4. Finally forward the `close()` call to the wrapped environment so
             that any other resources are released.
        """

        # 1. Tidy up the renderer managed by the wrapped environment (if any).
        base_env = self.env.unwrapped  # type: ignore[attr-defined]
        if hasattr(base_env, "_viewer"):
            viewer = base_env._viewer
            if viewer is not None and hasattr(viewer, "close") and callable(viewer.close):
                try:  # noqa: SIM105
                    viewer.close()
                except Exception:
                    # Ignore errors coming from older MuJoCo versions or
                    # already-freed contexts.
                    pass
            # Prevent the underlying env from trying to close it again.
            base_env._viewer = None

        # 2. Close the passive viewer launched by this wrapper.
        try:  # noqa: SIM105
            self._viewer.close()
        except Exception:  # pragma: no cover
            # Defensive: avoid propagating viewer shutdown errors.
            pass

        # 3. Let the wrapped environment perform its own cleanup.
        self.env.close()

    def __del__(self):
        # "close" may raise if called during interpreter shutdown; guard just
        # in case.
        if hasattr(self, "_viewer"):
            try:  # noqa: SIM105
                self._viewer.close()
            except Exception:
                pass

class KeyboardViewerWrapper(gym.Wrapper):
    """Gym wrapper that opens a passive MuJoCo viewer automatically.

    The wrapper starts a MuJoCo viewer in passive mode as soon as the
    environment is created so the user no longer needs to use
    ``mujoco.viewer.launch_passive`` or any context–manager boiler-plate.

    The viewer is kept in sync after every ``reset`` and ``step`` call and is
    closed automatically when the environment itself is closed or deleted.
    """

    def __init__(
        self,
        env: gym.Env,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        use_gripper=False,
        auto_reset=False,
        input_threshold=0.001,
        use_gamepad=True,
        controller_config_path=None,
        *,
        show_left_ui: bool = False,
        show_right_ui: bool = False,
    ) -> None:
        super().__init__(env)

        from gym_hil.wrappers.intervention_utils import (
            GamepadController,
            GamepadControllerHID,
            KeyboardController,
        )

        # use HidApi for macos
        if use_gamepad:
            if sys.platform == "darwin":
                self.controller = GamepadControllerHID(
                    x_step_size=x_step_size,
                    y_step_size=y_step_size,
                    z_step_size=z_step_size,
                )
            else:
                self.controller = GamepadController(
                    x_step_size=x_step_size,
                    y_step_size=y_step_size,
                    z_step_size=z_step_size,
                    config_path=controller_config_path,
                )
        else:
            self.controller = KeyboardController(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )

        self.auto_reset = auto_reset
        self.use_gripper = use_gripper
        self.input_threshold = input_threshold
        self.controller.start()

        # Launch the interactive viewer.  We expose *model* and *data* from the
        # *unwrapped* environment to make sure we operate on the base MuJoCo
        # objects even if other wrappers have been applied before this one.
        self._viewer = mujoco.viewer.launch_passive(
            env.unwrapped.model,
            env.unwrapped.data,
            # show_left_ui=show_left_ui,
            # show_right_ui=show_right_ui,
        )

        # Make sure the first frame is rendered.
        self._viewer.sync()

    # ---------------------------------------------------------------------
    # Gym API overrides

    def reset(self, **kwargs):  # type: ignore[override]
        observation, info = self.env.reset(**kwargs)
        self.controller.reset()
        self._viewer.sync()
        return observation, info
    
    def get_gamepad_action(self):
        """
        Get the current action from the gamepad if any input is active.

        Returns:
            Tuple of (is_active, action, terminate_episode, success)
        """
        # Update the controller to get fresh inputs
        self.controller.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.controller.get_deltas()

        intervention_is_active = self.controller.should_intervene()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        if self.use_gripper:
            gripper_command = self.controller.gripper_command()
            if gripper_command == "open":
                gamepad_action = np.concatenate([gamepad_action, [2.0]])
            elif gripper_command == "close":
                gamepad_action = np.concatenate([gamepad_action, [0.0]])
            else:
                gamepad_action = np.concatenate([gamepad_action, [1.0]])

        # Check episode ending buttons
        # We'll rely on controller.get_episode_end_status() which returns "success", "failure", or None
        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return (
            intervention_is_active,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        )


    def step(self, action):  # type: ignore[override]
        # Get gamepad state and action

        (
            is_intervention,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        # Update episode ending state if requested
        # if terminate_episode:
        #     logging.info(f"Episode manually ended: {'SUCCESS' if success else 'FAILURE'}")

        if is_intervention:
            action = gamepad_action
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._viewer.sync()

                # Add episode ending if requested via gamepad
        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0
            # logging.info("Episode ended successfully with reward 1.0")

        info["is_intervention"] = is_intervention
        action_intervention = action

        info["action_intervention"] = action_intervention
        info["rerecord_episode"] = rerecord_episode

        # If episode ended, reset the state
        if terminated or truncated:
            # Add success/failure information to info dict
            info["next.success"] = success

            # Auto reset if configured
            if self.auto_reset:
                observation, reset_info = self.reset()
                info.update(reset_info)

        return observation, reward, terminated, truncated, info

    def close(self) -> None:  # type: ignore[override]
        """Close both the passive viewer and the underlying gym environment.

        MuJoCo's `Renderer` gained a `close()` method only in recent versions
        (>= 2.3.0).  When running with an older MuJoCo build the renderer
        instance stored inside `env.unwrapped._viewer` does not provide this
        method which causes `AttributeError` when the environment is closed.

        To remain version-agnostic we:
          1. Manually dispose of the underlying viewer *only* if it exposes a
             `close` method.
          2. Remove the reference from the environment so that a subsequent
             call to `env.close()` will not fail.
          3. Close our own passive viewer handle.
          4. Finally forward the `close()` call to the wrapped environment so
             that any other resources are released.
        """

        # 1. Tidy up the renderer managed by the wrapped environment (if any).
        base_env = self.env.unwrapped  # type: ignore[attr-defined]
        if hasattr(base_env, "_viewer"):
            viewer = base_env._viewer
            if viewer is not None and hasattr(viewer, "close") and callable(viewer.close):
                try:  # noqa: SIM105
                    viewer.close()
                except Exception:
                    # Ignore errors coming from older MuJoCo versions or
                    # already-freed contexts.
                    pass
            # Prevent the underlying env from trying to close it again.
            base_env._viewer = None

        # 2. Close the passive viewer launched by this wrapper.
        try:  # noqa: SIM105
            self._viewer.close()
        except Exception:  # pragma: no cover
            # Defensive: avoid propagating viewer shutdown errors.
            pass
        
        """Clean up resources when environment closes."""
        # Stop the controller
        if hasattr(self, "controller"):
            self.controller.stop()

        # 3. Let the wrapped environment perform its own cleanup.
        self.env.close()

    def __del__(self):
        # "close" may raise if called during interpreter shutdown; guard just
        # in case.
        if hasattr(self, "_viewer"):
            try:  # noqa: SIM105
                self._viewer.close()
            except Exception:
                pass



