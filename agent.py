import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from dataclasses import dataclass
from torch.distributions.categorical import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pos_actions = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2],
               [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2],
               [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]
inter_actions = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def process(x):
    list = []
    for key in x.keys():
        value = x[key]
        value_shape = x[key].shape
        flat_shape = np.prod(value_shape)
        value_array = value.reshape((flat_shape,))
        list += value_array.tolist()
    flat_value_array = np.array(list)
    x = torch.Tensor(flat_value_array).to(device)
    return x


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 1000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


obs_size = 45
action_size = 6


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_size, 64)),
                                    nn.Tanh(),
                                    layer_init(nn.Linear(64, 64)),
                                    nn.Tanh(),
                                    layer_init(nn.Linear(64, 1), std=1.0))
        self.position_actor = nn.Sequential(layer_init(nn.Linear(obs_size, 64)),
                                            nn.Tanh(),
                                            layer_init(nn.Linear(64, 64)),
                                            nn.Tanh(),
                                            layer_init(nn.Linear(64, 27), std=0.01))
        self.interaction_actor = nn.Sequential(layer_init(nn.Linear(obs_size, 64)),
                                               nn.Tanh(),
                                               layer_init(nn.Linear(64, 64)),
                                               nn.Tanh(),
                                               layer_init(nn.Linear(64, 8), std=0.01))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        pos_logits = self.position_actor(x)
        inter_logits = self.interaction_actor(x)
        pos_probs = Categorical(logits=pos_logits)
        inter_probs = Categorical(logits=inter_logits)
        if action is None:
            pos_indexes = pos_probs.sample().item()
            inter_indexes = inter_probs.sample().item()
            pos_action = pos_actions[pos_indexes]
            inter_action = inter_actions[inter_indexes]
            action = pos_action + inter_action
            log_prob = torch.cat((pos_probs.log_prob(torch.Tensor(pos_action)),
                                  inter_probs.log_prob(torch.Tensor(inter_action)))).to(device)
            entropy = pos_probs.entropy() + inter_probs.entropy()
            return action, log_prob, entropy, self.critic(x)
        log_prob = torch.stack((pos_probs.log_prob(action[:, 0]).resize(250, 1),
                                pos_probs.log_prob(action[:, 1]).resize(250, 1),
                                pos_probs.log_prob(action[:, 2]).resize(250, 1),
                                inter_probs.log_prob(action[:, 3]).resize(250, 1),
                                inter_probs.log_prob(action[:, 4]).resize(250, 1),
                                inter_probs.log_prob(action[:, 5]).resize(250, 1)), dim=0).to(device)
        entropy = pos_probs.entropy() + inter_probs.entropy()
        return action, log_prob, entropy, self.critic(x)


class PPO:
    def __init__(self):
        self.args = Args()
        self.agent = Agent()
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = int(self.args.batch_size // self.args.num_minibatches)
        self.args.num_iterations = self.args.total_timesteps // self.args.batch_size

        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) + (obs_size, )).to(device)
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + (action_size, )).to(device)
        self.log_probs = torch.zeros((self.args.num_steps, self.args.num_envs, 6)).to(device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(device)

        self.next_obs = None
        self.next_done = torch.zeros(self.args.num_envs).to(device)

    def pre_step_train(self, step, next_obs):
        next_obs = process(next_obs)
        self.obs[step] = next_obs
        self.dones[step] = self.next_done
        with torch.no_grad():
            action, log_prob, _, value = self.agent.get_action_and_value(next_obs)
            self.values[step] = value.flatten()
        self.actions[step] = torch.FloatTensor(action)
        self.log_probs[step] = log_prob
        return action

    def post_step(self, step, next_obs, reward, done):
        next_obs = process(next_obs)
        next_done = done
        self.rewards[step] = torch.tensor(reward).to(device).view(-1)
        self.next_obs, self.next_done = next_obs, torch.Tensor([next_done]).to(device)

    def pre_step_eval(self, next_obs):
        next_obs = process(next_obs)
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(next_obs)
        return action

    def update(self):
        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        # flatten the batch
        b_obs = self.obs.reshape((-1,) + (obs_size, ))
        b_logprobs = self.log_probs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + (action_size, ))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                   b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -self.args.clip_coef,
                                                                self.args.clip_coef,)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
