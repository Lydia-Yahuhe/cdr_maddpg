import os
from copy import deepcopy

from torch.optim import Adam
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from algo.network import Critic, Actor
from algo.memory import ReplayMemory, Experience
from algo.misc import *


class MADDPG:
    def __init__(self, dim_obs, dim_act, discrete_list, args, record=True):
        self.args = args
        self.discrete_list = discrete_list
        self.GAMMA = args.gamma
        self.tau = args.tau
        self.var = 1.0

        self.actor = Actor(dim_obs, dim_act)
        self.critic = Critic(dim_obs, dim_act)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.c_lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.a_lr)

        self.memory = ReplayMemory(args.memory_length)
        self.batch_size = args.batch_size

        if record:
            self.writer = SummaryWriter('trained/logs/')
        else:
            self.writer = None
        self.c_loss, self.a_loss = [], []
        # net_visual([(1, dim_obs)], self.actor, 'actor')
        # net_visual([(1, 2, dim_obs), (1, 2, dim_act)], self.critic, 'critic')

    def load_model(self):
        print("load model!")
        actor = th.load(root + "actor.pth")
        critic = th.load(root + "critic.pth")
        self.actor.load_state_dict(actor.state_dict())
        self.critic.load_state_dict(critic.state_dict())
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def save_model(self):
        if not os.path.exists(root):
            os.mkdir(root)

        th.save(self.actor, root + 'actor.pth')
        th.save(self.critic, root + 'critic.pth')

    def update(self, step):
        transitions = self.memory.sample(self.batch_size)

        for n_agent, transition in transitions.items():
            batch = Experience(*zip(*transition))

            state_batch = th.stack(batch.states).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            next_states = th.stack(batch.next_states).type(FloatTensor)

            # 更新Critic
            self.actor.zero_grad()
            self.critic.zero_grad()
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            current_q = self.critic(state_batch, action_batch)
            # next_actions = gumbel_softmax(self.actor_target(next_states), self.discrete_list, noisy=False)
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = target_q * self.GAMMA + reward_batch

            q_loss = nn.MSELoss()(current_q, target_q.detach())
            q_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()
            self.c_loss.append(q_loss.detach().numpy())

            # 更新Actor
            self.actor.zero_grad()
            self.critic.zero_grad()
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # ac = gumbel_softmax(self.actor(state_batch), self.discrete_list, noisy=False)
            ac = self.actor(state_batch)

            actor_loss = -self.critic(state_batch, ac).mean()
            actor_loss.backward()
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optimizer.step()

            self.a_loss.append(actor_loss.detach().numpy())

        if step % 100 == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

            self.writer.add_scalar('c_loss', np.mean(self.c_loss), step)
            self.writer.add_scalar('a_loss', np.mean(self.a_loss), step)
            self.c_loss, self.a_loss = [], []

    def choose_action(self, states, noisy=True):
        # obs = th.from_numpy(np.stack(states)).float()
        # actions = self.actor(obs.detach()).detach()
        # actions = gumbel_softmax(actions, self.discrete_list, noisy=noisy, var=self.var)
        # if noisy and self.var > 0.05:
        #     self.var *= 0.99995
        # return actions.data.cpu().numpy()

        obs = th.from_numpy(np.stack(states)).float()
        actions = self.actor(obs.detach())

        if noisy:
            actions += th.randn(actions.shape)*self.var

        actions = th.clamp(actions, -1, 1)
        if self.var > 0.05:
            self.var *= 0.9999

        return actions.data.cpu().numpy()

    def scalars(self, key, value, episode):
        self.writer.add_scalars(key, value, episode)

    def scalar(self, key, value, episode):
        self.writer.add_scalar(key, value, episode)

    def close(self):
        if self.writer is not None:
            self.writer.close()
