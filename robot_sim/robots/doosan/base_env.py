
    
    
from typing import Optional
import gymnasium as gym 
import numpy as np

class BaseRobotArmEnv(gym.Env):
    def __init__(self):
        self.reward_type = 'sparse'
        self.goal_threshold = 0.02
        self.target_range = 0.5
        
    def get_dist(self, achieved_goal, desired_goal):
        return np.linalg.norm(achieved_goal - desired_goal)
        
    def compute_reward(self, achieved_goal, desired_goal, info):

        ag = np.asarray(achieved_goal, dtype=np.float32)
        dg = np.asarray(desired_goal,  dtype=np.float32)

        dist = np.linalg.norm(ag - dg, axis=-1)

        if getattr(self, "reward_type", "dense") == "dense":
            r = -dist
        else:
            thr = float(getattr(self, "goal_threshold", 0.02))
            r = -(dist > thr).astype(np.float32) 
        
        return r.astype(np.float32) # 스칼라로 들어오면 스칼라(float32)로, 배치면 (N,) float32로 반환

    def sample_goal(self):
        goal = self.init_ee_position + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
        
        return goal
        
    def _get_obs(self):
        obs = {
            'achieved_goal': self.curr_ee_position,
            'desired_goal': self.goal,
            'observations': list(np.concatenate([self.curr_joint_position, np.array([e - g for e, g in zip(self.curr_ee_position, self.goal)])])),
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
    

if __name__ == '__main__':
    import time
    
    gym.register(
        id="M1013Env-v0",
        entry_point=M1013Env,
        max_episode_steps=100,  # Prevent infinite episodes
    )

    env_name = 'M1013Env-v0'
    env = gym.make(env_name)

    obs, info = env.reset()
    for _ in range(10):
        tic = time.time() 
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        print("Time: ", time.time() - tic)
        dist = np.linalg.norm(obs['achieved_goal']- obs['desired_goal'])
        print(obs)
        print(reward)
        print(dist)
    