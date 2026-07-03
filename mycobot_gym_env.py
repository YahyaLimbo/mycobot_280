import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import time

class MyCobotGymEnv(gym.Env):
    """
    A Gymnasium wrapper for the MyCobot 280 in Gazebo/ROS 2.
    This class handles ROS 2 communication and wraps it into a standard RL interface.
    """
    
    def __init__(self):
        super(MyCobotGymEnv, self).__init__()

        # 1. --- ROS 2 Initialization ---
        rclpy.init()
        self.node = Node('mycobot_gym_env_node')

        # 2. --- Define Action and Observation Spaces ---
        self.num_joints = 6
        self.action_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32
        )

        # Observation: position (6) + velocity (6) = 12 values
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_joints * 2,), dtype=np.float32
        )

        # 3. --- ROS 2 Communication Setup ---
        # Replace this topic with your actual controller command topic!
        self.cmd_pub = self.node.create_publisher(
            Float64MultiArray, '/mycobot_280/joint_trajectory_controller/commands', 10
        )

        # Subscriber to get joint states
        self.current_state = None
        self.cmd_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

        self.target_position = np.zeros(self.num_joints)
        self.steps_taken = 0
        self.max_steps = 200 

    def _joint_state_callback(self, msg):
        if len(msg.position) >= self.num_joints:
            pos = msg.position[:self.num_joints]
            vel = msg.velocity[:self.num_joints]
            self.current_state = np.concatenate([pos, vel]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.node.get_logger().info("Resetting Environment...")
        
        # Randomize target position for training variety
        self.target_position = np.random.uniform(-np.pi/2, np.pi/2, self.num_joints)

        # Reset robot to zero postion (as a simple start)
        self._send_action(np.zeros(self.num_joints))

        # Wait for the first valid observation from the subscriber
        timeout = 5.0
        start_time = time.time()
        while self.current_state is None and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            rclpy.spin_once(self.node, timeout_sec=0.01)

        self.steps_taken = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.steps_taken += 1
        
        # Apply Action (send joint targets)
        self._send_action(action)

        # Wait for physics to update
        time.sleep(0.1) 
        rclpy.spin_once(self.node, timeout_sec=0.05)

        obs = self._get_obs()
        
        # Reward: negative Euclidean distance to target position
        current_pos = obs[:self.num_joints]
        distance_error = np.linalg.norm(current_pos - self.target_position)
        reward = -float(distance_error)

        terminated = False 
        truncated = self.steps_taken >= self.max_steps

        return obs, reward, terminated, truncated, {"error": distance_error}

    def _send_action(self, action):
        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.cmd_pub.publish(msg)

    def _get_obs(self):
        if self.current_state is None:
            return np.zeros(self.num_joints * 2, dtype=np.float32)
        return self.current_state

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    # Test Execution
    try:
        env = MyCobotGymEnv()
        obs, info = env.reset()
        print("Initial Observation:", obs)
        for i in range(10):
            action = env.action_space.sample() # Random actions
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Reward={reward:.4f}, Error={info['error']:.4f}")
            if terminated or truncated:
                break
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        env.close()
