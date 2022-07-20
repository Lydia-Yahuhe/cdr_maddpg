import argparse

from flightEnv.env import ConflictEnv

from algo.maddpg_agent import MADDPG
from algo.misc import *


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", default='exp_0', type=str)

    parser.add_argument('--max_episodes', default=int(1e6), type=int)
    parser.add_argument('--max_step_per_epi', default=10, type=int)
    parser.add_argument('--memory_length', default=int(5e4), type=int)

    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.001, type=float)
    parser.add_argument('--c_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument("--save_interval", default=5000, type=int)
    parser.add_argument('--step_before_train', default=100, type=int)

    return parser.parse_args()


def train():
    args = args_parse()
    th.manual_seed(args.seed)

    env = ConflictEnv(x=0, size=1, ratio=1.0)
    model = MADDPG(env.observation_space.shape[0], env.action_space.n,
                   env.discrete_list, args)
    # model.load_model()

    step = 0
    rew_epi, solved_step, solved_epi = [], [], []
    step_epi, mode = [], 'next_p'
    for episode in range(1, args.max_episodes):
        states, t = env.reset(mode=mode), 0

        while True:
            actions = model.choose_action(states, noisy=True)
            next_states, rewards, done, info = env.step(actions)
            t += 1
            step += 1
            # env.render(counter='{}_{}_{}'.format(t, step, episode))

            # replay buffer R
            obs = th.from_numpy(np.stack(states)).float().to(device)
            next_obs = th.from_numpy(np.stack(next_states)).float().to(device)
            rw_tensor = th.FloatTensor(np.array([sum(rewards)])).to(device)
            ac_tensor = th.FloatTensor(actions).to(device)
            model.memory.push(obs.data, ac_tensor, next_obs.data, rw_tensor)
            # states = next_states

            solved_step.append(float(done))
            rew_epi.append(sum(rewards))
            print('t:{:>2d}, step:{:>6d}, episode:{:>6d}'.format(t, step, episode), end='\t')
            print(['{:>+4.2f}'.format(rew) for rew in rewards])

            if step >= args.step_before_train:
                model.update(step)

            if done or t >= args.max_step_per_epi:
                mode = 'next_p' if done else 'next_s'
                step_epi.append(t)
                break

        solved_epi.append(float(done))

        if episode % 10 == 0:
            model.scalar("reward", np.mean(rew_epi), episode)
            model.scalars("sr", {'step': np.mean(solved_step),
                                 'episode': np.mean(solved_epi)}, episode)
            model.scalar("times", np.mean(step_epi), episode)

            rew_epi, solved_step, solved_epi, step_epi = [], [], [], []
            if episode % args.save_interval == 0:
                model.save_model()

    model.close()


if __name__ == '__main__':
    train()
