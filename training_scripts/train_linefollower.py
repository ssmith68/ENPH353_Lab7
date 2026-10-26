#!/usr/bin/env python3
import random
import gym_gazebo
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

def main():
    writer = SummaryWriter()
    env = gym.make('LineFollowerEnv-v0')

    observation_space = env.observation_space
    action_space = env.action_space

    epsilon = 0.8
    epsilon_decay = 0.99

    q_table = np.zeros((observation_space.n, action_space.n))

    highest_reward = 0

    try:
        state, _ = env.reset()
        for episode in range(1000):
            total_reward = 0
            actions_per_episode = 0
            done = False

            while not done:
                actions_per_episode += 1
                action = choose_action(state, q_table, epsilon, action_space)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated or truncated

                q_table = learn(state, action, next_state, reward, q_table)

                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            print("Q values:")
            print(q_table)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)

            if total_reward > highest_reward:
                highest_reward = total_reward
                print("New highest reward: {total_reward}.")
                saveQ(q_table, "best_q_table.csv")

            env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    finally:
        env.close()
        writer.flush()
        writer.close()


def learn(state, action, next_state, reward, q_table):
    alpha = 0.1
    gamma = 0.9

    # Get the current Q-value for the state-action pair
    current_q = q_table[state, action]
    
    # Get the maximum future reward for the next state.
    # If the episode terminated (next_state == -1), there is no future reward.
    if next_state == -1:
        max_next_q = 0.0
    else:
        max_next_q = np.max(q_table[next_state])
        
    # Apply the Bellman update equation
    new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
    
    # Update the Q-table with the new value
    q_table[state, action] = new_q

    return q_table


def choose_action(state, q_table, epsilon, action_space):
    # Generate a random number to decide between exploration and exploitation
    if np.random.random() < epsilon:
        # EXPLORE: Choose a random action
        action = action_space.sample() 
    else:
        # EXPLOIT: Choose the best action based on Q-values
        state_q_values = q_table[state]
        max_q = np.max(state_q_values)
        
        # Find all actions that share the maximum Q-value (handles ties)
        best_actions = np.where(state_q_values == max_q)[0]
        
        # Break ties randomly among the best actions
        action = np.random.choice(best_actions)
        
    return int(action)


def saveQ(q_table, save_path):
    # Save the numpy array as a comma-separated values file
    np.savetxt(save_path, q_table, delimiter=',')
    print(f"--> Saved best Q-table to {save_path}")

def loadQ(load_path):
    # Load the numpy array from the CSV file
    try:
        q_table = np.loadtxt(load_path, delimiter=',')
        print(f"--> Successfully loaded Q-table from {load_path}")
        return q_table
    except FileNotFoundError:
        print(f"--> File {load_path} not found. Starting with a fresh Q-table.")
        # If the file doesn't exist yet, we return None so main() knows to start fresh
        return None

if __name__ == "__main__":
    main()