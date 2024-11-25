import sys
import numpy as np
from zmqRemoteApi import RemoteAPIClient
import time
import os

class Simulation():
    def __init__(self, sim_port=23004):
        self.sim_port = sim_port
        self.directions = ['Up', 'Down', 'Left', 'Right']
        self.initializeSim()

    def initializeSim(self):
        self.client = RemoteAPIClient('localhost',port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')
        
        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)  
        
        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
        self.getObjectsInBoxHandles()
    
    def getObjectHandles(self):
        self.tableHandle=self.sim.getObject('/Table')
        self.boxHandle=self.sim.getObject('/Table/Box')
    
    def dropObjects(self):
        self.blocks = 18
        frictionCube=0.06
        frictionCup=0.8
        blockLength=0.016
        massOfBlock=14.375e-03
        
        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript,self.tableHandle)
        self.client.step()
        retInts,retFloats,retStrings=self.sim.callScriptFunction('setNumberOfBlocks',self.scriptHandle,[self.blocks],[massOfBlock,blockLength,frictionCube,frictionCup],['cylinder'])
        
        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signalValue=self.sim.getFloatSignal('toPython')
            if signalValue == 99:
                loop = 20
                while loop > 0:
                    self.client.step()
                    loop -= 1
                break
    
    def getObjectsInBoxHandles(self):
        self.object_shapes_handles=[]
        self.obj_type = "Cylinder"
        for obj_idx in range(self.blocks):
            obj_handle = self.sim.getObjectHandle(f'{self.obj_type}{obj_idx}')
            self.object_shapes_handles.append(obj_handle)

    def getObjectsPositions(self):
        pos_step = []
        box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        for obj_handle in self.object_shapes_handles:
            # get the starting position of source
            obj_position = self.sim.getObjectPosition(obj_handle,self.sim.handle_world)
            obj_position = np.array(obj_position) - np.array(box_position)
            pos_step.append(list(obj_position[:2]))
        return pos_step
    
    def action(self,direction=None):
        if direction not in self.directions:
            print(f'Direction: {direction} invalid, please choose one from {self.directions}')
            return
        box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        _box_position = box_position
        span = 0.01
        steps = 3
        if direction == 'Up':
            idx = 1
            dirs = [1, -1]
        elif direction == 'Down':
            idx = 1
            dirs = [-1, 1]
        elif direction == 'Right':
            idx = 0
            dirs = [1, -1]
        elif direction == 'Left':
            idx = 0
            dirs = [-1, 1]

        for _dir in dirs:
            for _ in range(steps):
                _box_position[idx] += _dir*span / steps
                self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
                self.stepSim()

    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()
class QLearningBlockSimulation(Simulation):
    def __init__(self, sim_port=23004, learning_rate=0.2, discount_factor=0.8, exploration_prob=0.4):
        super().__init__(sim_port)
        self.alpha_rate = learning_rate
        self.beta_factor = discount_factor
        self.explore_prob = exploration_prob
        self.initialize_qtable()

    def initialize_qtable(self):
        self.qvalues_matrix = np.zeros((2 * len(self.object_shapes_handles), len(self.directions)))

    def train_qlearning_episodes(self, num_episodes=10, num_steps=20):
        total_episodes_rewards = []

        for episode in range(1, num_episodes + 1):
            print(f'Running episode: {episode}')

            episode_reward = self.run_single_qlearning_episode(num_steps)

            print(f'Total Reward for Episode {episode}: {episode_reward}')
            self.record_trainingandtesting_rewards(episode_reward, episode, 'reward_on_train.txt')
            total_episodes_rewards.append(episode_reward)

        print("Training finished for the Q learing \nTotal Rewards:", total_episodes_rewards)
        print("Q-table:\n", self.qvalues_matrix)
        print('Training completed for the Q learning algorithm.')

    def test_qlearning_episodes(self, num_episodes=10, num_steps=20, threshold=0.8):
        total_episodes_rewards = []
        start_time = time.time()
        successful_trials = 0

        for episode in range(1, num_episodes + 1):
            print(f'Running episode: {episode}')

            episode_reward = self.run_single_qlearning_episode(num_steps)

            print(f'Total Reward for Episode {episode}: {episode_reward}')
            total_episodes_rewards.append(episode_reward)

            if episode_reward > threshold:
                successful_trials += 1

        end_time = time.time()
        elapsed_time = end_time - start_time

        self.log_elapsed_time(elapsed_time, num_episodes, 'time.txt')
        self.log_successful_trials(num_episodes, successful_trials)

        print("Testing finished for Q learning .\nTotal Rewards:", total_episodes_rewards)
        print("Q-table:\n", self.qvalues_matrix)
        print('Testing completed for Q learning .')

    def run_single_qlearning_episode(self, num_steps):
        episode_reward = 0
        current_state = self.getObjectsPositions()
        print("Initial State:", current_state)

        for step in range(num_steps):
            current_action = self.choose_action(current_state)

            next_state, reward = self.perform_step_and_calculating_reward(current_action, current_state)
            episode_reward += reward

            self.update_qvalues_matrix(self.calculating_state_index(current_state),
                                        self.calculating_action_index(current_action),
                                        reward,
                                        self.calculating_state_index(next_state))

            current_state = next_state

        return episode_reward

    def log_elapsed_time(self, elapsed_time, episode_number, file_name='time.txt'):
        with open(file_name, 'a') as file:
            file.write(f'For Episode {episode_number}: Total time taken for running in Coppelia Sim: {elapsed_time} seconds\n')

    def choose_action(self, state):
        return np.random.choice(self.directions) if np.random.uniform(0, 1) < self.explore_prob else self.select_action(state)

    def calculating_state_index(self, current_state):
        self.state_space_size = 2 * len(self.object_shapes_handles)
        state_index = hash(tuple(tuple(pos) for pos in current_state)) % self.state_space_size
        return state_index

    def calculating_action_index(self, current_action):
        action_mapping = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3}
        return action_mapping.get(current_action, None)

    def perform_step_and_calculating_reward(self, chosen_action, current_state):
        self.action(direction=chosen_action)
        updated_state = self.getObjectsPositions()
        target = 0.5
        tolerance = 0.5
        difference = abs(np.mean(current_state) - target)
        reward = 1.0 if difference < tolerance else 0.0
        return updated_state, reward

    def record_trainingandtesting_rewards(self, episode_reward, episode_number, file_path='reward_on_train.txt'):
        file_type_prefix = 'Test' if 'test' in file_path.lower() else ''
        with open(file_path, 'a') as output_file:
            output_file.write(f'Episode {episode_number}: Reward {episode_reward} {file_type_prefix}\n')

    def log_successful_trials(self, trials, successful_trials, file_name='time.txt'):
        with open(file_name, 'a') as file:
            file.write(f'Successful Trials for {trials} trials: {successful_trials}/{trials}\n')

    def clear_log_files(self, *file_names):
        for file_name in file_names:
            if os.path.exists(file_name):
                open(file_name, 'w').close()

    def write_qtable_to_file(self, file_name='q_table.txt'):
        with open(file_name, 'w') as file:
            file.write(f'Q-table after Training:\n')
            file.write(np.array_str(self.qvalues_matrix))

    def update_qvalues_matrix(self, state_index, action_index, reward, next_state_index):
        current_q_value = self.qvalues_matrix[state_index, action_index]
        max_future_q_value = np.max(self.qvalues_matrix[next_state_index])
        new_q_value = current_q_value + self.alpha_rate * (reward + self.beta_factor * max_future_q_value - current_q_value)
        self.qvalues_matrix[state_index, action_index] = new_q_value

    def select_action(self, state):
        state_index = self.calculating_state_index(state)
        qvalues = self.qvalues_matrix[state_index]
        selected_action_index = np.argmax(qvalues)
        return self.directions[selected_action_index]

    def run_random_episodes(self, num_episodes=10, num_steps=20):
        elapsed_time = 0
        successful_trials = 0
        total_episodes_rewards = []

        for episode in range(1, num_episodes + 1):
            print(f'Running random episode: {episode}')

            begin_time = time.time()

            episode_reward = 0
            current_state = self.getObjectsPositions()
            print("Initial State:", current_state)

            for step in range(num_steps):
                current_action = np.random.choice(self.directions)
                next_state, reward = self.perform_step_and_calculating_reward(current_action, current_state)
                episode_reward += reward
                current_state = next_state

            end_time = time.time()
            elapsed_time += end_time - begin_time

            print(f'Total Reward for Random Episode {episode}: {episode_reward}')
            total_episodes_rewards.append(episode_reward)

            if episode_reward > 0:
                successful_trials += 1

            self.record_trainingandtesting_rewards(episode_reward, episode, 'reward_if_random.txt')

        self.log_elapsed_time(elapsed_time, num_episodes, 'time_if_random.txt')
        self.log_successful_trials(num_episodes, successful_trials, 'time_if_random.txt')

if __name__ == '__main__':
    q_learning_env = QLearningBlockSimulation()
    q_learning_env.clear_log_files('time.txt', 'reward_on_train.txt', 'reward_if_random.txt', 'time_if_random.txt')

    # Training with Q-learning
    q_learning_env.train_qlearning_episodes(num_episodes=10, num_steps=20)
    q_learning_env.write_qtable_to_file('q_table.txt')

    # Testing with Q-learning
    q_learning_env.test_qlearning_episodes(num_episodes=10, num_steps=20, threshold=0.6)

    # Running random episodes
    q_learning_env.run_random_episodes(num_episodes=10, num_steps=20)

    q_learning_env.stopSim()
    time.sleep(1)
