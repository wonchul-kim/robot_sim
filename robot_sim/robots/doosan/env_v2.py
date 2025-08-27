import numpy as np
from typing import Optional
import gymnasium as gym 
import numpy as np
from robot_sim.robots.doosan import Kinematics

class M1013EnvV2(gym.Env):
    def __init__(self):
        
        self.k = Kinematics('/HDD/_projects/github/robot/data/descriptions/doosan/m1013_modified_v2.urdf')
        
        self.init_joint_position = [0.0, -0.17444443702697754, 1.5700000524520874, 0.0, 1.5700000524520874, 0.0]
        self.curr_joint_position = self.init_joint_position
        self.init_ee_position = [0.4676724374294281, 0.03708246722817421, 0.7366037368774414]
        self.curr_ee_position = self.init_ee_position
        self.current_step = 1
        self.reward_type = 'dense'
        self.target_range = 0.5
        self.goal_threshold = 0.02
        self.action_scale = 0.05
        self.max_steps_per_episode = 100
        
        self.observation_space = gym.spaces.Dict(
            {
                "achieved_goal": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=float), 
                "desired_goal": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=float),
                "observations": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=float,
                )
            }
        )
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3 ,), dtype=float)
        self.goal = None
        
    def get_dist(self, achieved_goal, desired_goal):
        return np.linalg.norm(achieved_goal - desired_goal)
        
    def compute_reward(self, achieved_goal, desired_goal, info):

        ag = np.asarray(achieved_goal, dtype=np.float32)
        dg = np.asarray(desired_goal,  dtype=np.float32)

        dist = np.linalg.norm(ag - dg, axis=-1)

        if getattr(self, "reward_type", "dense") == "dense":
            r = -dist
        else:  # 'sparse'
            thr = float(getattr(self, "goal_threshold", 0.02))
            r = -(dist > thr).astype(np.float32)  # <=thr -> 0,  >thr -> -1

        # 스칼라로 들어오면 스칼라(float32)로, 배치면 (N,) float32로 반환
        return r.astype(np.float32)

    def sample_goal(self):
        goal = self.init_ee_position + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
        
        return goal
        
    def _get_obs(self):
        obs = {
            'achieved_goal': self.curr_ee_position,
            'desired_goal': self.goal,
            'observations': self.curr_ee_position,
        }
        return obs 
    
    def _get_info(self):
        
        return {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.curr_joint_position = self.init_joint_position
        self.curr_ee_position = self.init_ee_position

        self.goal = self.sample_goal()
        observation = self._get_obs()
        info = self._get_info()
        self.current_step = 1
        
        return observation, info 
    
    def step(self, action):
        # Map the discrete action (0-3) to a movement direction
        
        assert len(action) == self.action_space.shape[0]
        self.curr_ee_position += action*self.action_scale
        target = np.vstack([np.hstack([np.eye(3), self.curr_ee_position.reshape(-1, 1)]), [0, 0, 0, 1]])
        self.curr_joint_position = self.k.ik_frame(target=target, 
                                                   initial_position=np.concatenate(([0], self.curr_joint_position, [0])))

        observation = self._get_obs()
        info = self._get_info()

        terminated = False
        if self.get_dist(self.curr_ee_position, self.goal) < float(getattr(self, "goal_threshold", 0.02)):
            terminated = True 
            info.update({"is_success": True})
        else:
            info.update({"is_success": False})
            
        truncated = False
        if self.current_step >= self.max_steps_per_episode:
            truncated = True 
        self.current_step += 1
        
        if getattr(self, "reward_type", "dense") == "dense":
            reward = np.array([[0]]) if terminated else np.array([[-self.get_dist(self.curr_ee_position, self.goal)]])
        else:  # 'sparse'
            reward = np.array([[0]]) if terminated else np.array([[-1]])
        
        return observation, reward, terminated, truncated, info        
        

if __name__ == '__main__':
    import time
    gym.register(
        id="M1013Env-v1",
        entry_point=M1013EnvV2,
        max_episode_steps=100,  # Prevent infinite episodes
    )

    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)

    env_name = 'M1013Env-v1'
    # env_name = 'FetchReach-v4'
    env = gym.make(env_name)

    obs, info = env.reset()
    for _ in range(10):
        tic = time.time()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print("time: ", time.time() - tic)
        dist = np.linalg.norm(obs['achieved_goal']- obs['desired_goal'])
        print('action: ', action)
        print(obs)
        print(reward)
        print(dist)
    