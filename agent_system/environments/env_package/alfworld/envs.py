# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import copy
import os

import gymnasium as gym
import ray
import torch
import torchvision.transforms as T
import yaml

from agent_system.alfworld_game_manifest import (
    WORKSTREAM_B_INFO_KEYS,
    canonicalize_schedule,
    load_manifest,
)
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.base_env = base_env
        self.seed = seed
        self.env = self._make_env()
        self.current_schedule = None

    def _make_env(self, gamefile=None):
        env_factory = copy.copy(self.base_env)
        if gamefile is not None:
            env_factory.game_files = [gamefile]
            env_factory.num_games = 1
        env = env_factory.init_env(batch_size=1)
        env.seed(self.seed)
        return env

    def _select_scheduled_game(self, schedule):
        gamefile = schedule["wm_gamefile"]
        if self.current_schedule is None or self.current_schedule["wm_gamefile"] != gamefile:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
            self.env = self._make_env(gamefile=gamefile)
        self.current_schedule = dict(schedule)

    def _attach_schedule(self, infos, verify_actual=False):
        if self.current_schedule is None:
            return infos
        actual_gamefiles = infos.get("extra.gamefile")
        actual_gamefile = actual_gamefiles[0] if actual_gamefiles is not None and len(actual_gamefiles) else None
        if verify_actual and os.path.realpath(str(actual_gamefile)) != self.current_schedule["wm_gamefile"]:
            raise RuntimeError(
                "ALFWorld scheduled game mismatch: "
                f"expected={self.current_schedule['wm_gamefile']!r} actual={actual_gamefile!r}"
            )
        if actual_gamefile is not None and os.path.realpath(str(actual_gamefile)) != self.current_schedule["wm_gamefile"]:
            raise RuntimeError(
                "ALFWorld game changed during scheduled episode: "
                f"expected={self.current_schedule['wm_gamefile']!r} actual={actual_gamefile!r}"
            )
        infos["extra.gamefile"] = [self.current_schedule["wm_gamefile"]]
        for key in WORKSTREAM_B_INFO_KEYS:
            infos[key] = [self.current_schedule[key]]
        return infos
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        infos = self._attach_schedule(infos)
        return obs, scores, dones, infos
    
    def reset(self, schedule=None):
        """Reset the environment"""
        if schedule is not None:
            self._select_scheduled_game(schedule)
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        infos = self._attach_schedule(infos, verify_actual=True)
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None):
        super().__init__()
        env_kwargs = env_kwargs or {}
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.require_manifest_schedule = bool(env_kwargs.get("require_manifest_schedule", False))
        manifest_path = env_kwargs.get("manifest_path")
        self.manifest = None
        if manifest_path:
            self.manifest = load_manifest(
                manifest_path,
                expected_games=env_kwargs.get("manifest_expected_games"),
                expected_raw_trajectories=env_kwargs.get(
                    "manifest_expected_raw_trajectories"
                ),
                require_train=True,
                verify_files=bool(env_kwargs.get("verify_manifest_files", True)),
            )
        if self.require_manifest_schedule and self.manifest is None:
            raise ValueError("require_manifest_schedule=true requires env.alfworld.manifest_path")

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def _normalise_schedule(self, schedule):
        return canonicalize_schedule(
            self.manifest,
            schedule,
            env_num=self.env_num,
            group_n=self.group_n,
            require_schedule=self.require_manifest_schedule,
        )

    def reset(self, schedule=None):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        schedule_entries = self._normalise_schedule(schedule)
        for worker, schedule_entry in zip(self.workers, schedule_entries):
            future = worker.reset.remote(schedule_entry)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)