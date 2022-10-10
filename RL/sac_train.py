import argparse
import datetime
import gym
import numpy as np
import os
import itertools
import torch
from pathlib import Path
from RL.pytorch_soft_actor_critic.sac import SAC
from RL.pytorch_soft_actor_critic.utils import *
from torch.utils.tensorboard import SummaryWriter
from RL.pytorch_soft_actor_critic.replay_memory import ReplayMemory
from RL.DIP_env import double_inverted_pendulum


def set_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="Pendulum-v0",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    args = parser.parse_args("")

    return args


def save_dyn_dict(state, action):
    state_arr = []
    pass


def train_sac(env, agent, writer, args):
    # Training Loop
    total_numsteps = 0
    updates = 0
    # Memory
    memory = ReplayMemory(args.replay_size)
    for i_episode in range(1, args.max_episodes + 1):
        # for i_episode in itertools.count(1): # add max episodes

        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:

            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         args.batch_size,
                                                                                                         updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)

                    updates += 1
            env.render()
            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state
            writer.add_scalar('states/phi_1', state[2], i_episode)
            writer.add_scalar('states/phi_2', state[4], i_episode)
            writer.add_scalar('states/x', state[0], i_episode)
        # if total_numsteps > args.num_steps:
        #     break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))

        if i_episode % 200 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False

                while not done:
                    env.render()
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward

                    state = next_state

                avg_reward += episode_reward

            avg_reward /= episodes

            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

        if i_episode % 100 == 0:
            state = env.reset()
            episode_reward = 0
            done = False
            # save to render
            save_render_dict = {}
            states_list = []
            actions_list = []
            rewards_list = []

            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
                states_list.append(state)
                actions_list.append(action)
                rewards_list.append(episode_reward)

            save_render_dict[i_episode] = {}
            save_render_dict[i_episode]["states"] = states_list
            save_render_dict[i_episode]["actions"] = actions_list
            save_render_dict[i_episode]["rewards"] = rewards_list

            print("----------------------------------------")
            print("Test Episodes: {}, save states and action".format(i_episode))
            print("----------------------------------------")
            torch.save(save_render_dict,
                       os.path.join(results_folder, str(i_episode) + 'eval_dynamics.pth.tar'))

    env.close()


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "runs"
    # Hyperparameters
    param_dict = {
        "policy": "Gaussian",
        "gamma": 0.99,
        "tau": 0.005,
        "lr": 0.003,
        "alpha": 0.6,
        "automatic_entropy_tuning": True,
        "batch_size": 256,
        "num_steps": 10000,
        "hidden_size": 256,
        "updates_per_step": 1,
        "cuda": True,
        "eval": True,
        "max_episodes": 1000}

    args = set_args()
    update_args(args, param_dict)
    results_folder = os.path.join(output_folder, "sac_new_swing_up",
                                  datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f"))
    try:
        os.makedirs(results_folder)
    except OSError:
        pass

    torch.save(args, os.path.join(results_folder, 'args.pth.tar'))
    # Environment
    maneuver = "swing_up"
    env = double_inverted_pendulum(maneuver=maneuver)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Memory
    memory = ReplayMemory(args.replay_size)

    # TesnorboardX
    writer = SummaryWriter(
        log_dir=os.path.join(results_folder, '{}_SAC_x_threshold_{}_max_episode_steps{}_input_{}_{}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            env.x_threshold, env._max_episode_steps, env.action_space.high.item(), maneuver)))

    train_sac(env, agent, writer, args)
