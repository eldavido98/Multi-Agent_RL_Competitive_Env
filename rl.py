from operator import itemgetter
from agent import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obs_process(x):
    dict_1, dict_2, dict_3, dict_4 = {}, {}, {}, {}
    dict_list = [dict_1, dict_2, dict_3, dict_4]
    for i in range(4):
        dict_list[i]['agent'] = x['agent'][i]
        dict_list[i]['others'] = x['others'][i]
        dict_list[i]['others_mask'] = x['others_mask'][i]
        dict_list[i]['zone'] = x['zone'][i]
    return dict_list


class PBT_MARL(nn.Module):
    def __init__(self, env, R_init=0, num_agents=32, K=0.1, T_select=0.47):
        # K : step size of Elo rating update given one match result
        # T_select : win rate selection threshold below which A_i should evolve to A_j
        super(PBT_MARL, self).__init__()
        self.device = device
        self.env = env
        self.R_init = R_init
        self.beginning_training_threshold = 500
        self.last_eligible_threshold = 500
        self.frames_since_beginning = 0
        self.frames_since_last_eligible = 0
        self.frames_since_last_evolution = 0
        self.num_agents = num_agents
        self.num_training_fights = int(num_agents/2)
        self.fight = Experiment(env, 500)
        self.agents = []
        for n in range(self.num_agents):
            agent = PPO()
            self.agents.append([agent, self.R_init, self.frames_since_beginning, self.frames_since_last_eligible,
                                self.frames_since_last_evolution, n])
        self.K = K
        self.T_select = T_select

    def train(self, max_episodes=5):
        done = True
        count = 0
        training_fights = []
        print("Start of training")
        while done:
            print("Episode ", count)
            for train in range(self.num_training_fights):
                i, j, k, l = np.random.random_integers(low=0, high=(self.num_agents-1), size=4)
                s_i_j, s_k_l = self.fight.training_fight([self.agents[i], self.agents[j], self.agents[k],
                                                          self.agents[l]])
                training_fights.append([i, j, k, l, s_i_j, s_k_l])
            for n in range(len(training_fights)):
                self.UpdateRating(training_fights[n][0], training_fights[n][1], training_fights[n][2],
                                  training_fights[n][3], training_fights[n][4], training_fights[n][5])

            for i in range(len(self.agents)):
                if self.Eligible(i):                                    # Check the conditions
                    agent_j = self.Select(i)
                    if agent_j is not None:
                        self.Inherit(i, agent_j)
                        self.mutate(i)
                        self.agents[i][4] = 0

            if count == max_episodes:
                done = False
            count += 1

    def UpdateRating(self, i, j, k, l, s_i_j, s_k_l):
        rating_i, rating_j, rating_k, rating_l = self.agents[i][1], self.agents[j][1],\
                                                 self.agents[k][1], self.agents[l][1]
        # Avg Elo score
        s = (np.sign(s_i_j - s_k_l) + 1) / 2
        mean_r_i_j = np.mean(np.array([rating_i, rating_j]))
        mean_r_k_l = np.mean(np.array([rating_k, rating_l]))
        exp = (mean_r_i_j - mean_r_k_l) / 400
        s_elo = 1 / (1 + 10**exp)

        self.agents[i][1] = self.agents[i][1] + self.K * (s - s_elo)
        self.agents[j][1] = self.agents[j][1] + self.K * (s - s_elo)
        self.agents[k][1] = self.agents[k][1] + self.K * (s - s_elo)
        self.agents[l][1] = self.agents[l][1] + self.K * (s - s_elo)

    def Eligible(self, i):
        # Check if the agent has processed at least 500 frames for learning since the beginning of training.
        if (self.agents[i][2] > self.beginning_training_threshold) and (self.agents[i][3] is None):
            self.agents[i][3] = 0
            return True
        # Check if the agent has processed at least 500 frames for learning since the last time it became eligible for evolution.
        if (self.agents[i][2] > self.beginning_training_threshold) and\
                (self.agents[i][3] > self.last_eligible_threshold):
            self.agents[i][3] = 0
            return True

        return False

    def Select(self, i):
        j = np.random.choice(set(range(self.num_agents)).remove(i))
        if (self.agents[j][4] is None) or (self.agents[j][4] < self.last_eligible_threshold):
            return None
        exp = (self.agents[j][1] - self.agents[i][1]) / 400
        s_elo = 1 / (1 + 10 ** exp)                                         # normalized between 0 and 1
        if s_elo < self.T_select:
            return self.agents[j]
        else:
            return None

    def Inherit(self, i, agent_j):
        self.agents[i][0].agent.parameters = agent_j[0].agent.parameters()
        m = np.random.binomial(n=1, p=0.5)
        # learning rate
        self.agents[i][0].args.learning_rate = m * self.agents[i][0].args.learning_rate + (1 - m) * agent_j[0].args.learning_rate
        self.agents[i][0].optimizer.learning_rate = self.agents[i][0].args.learning_rate
        # gamma
        self.agents[i][0].args.gamma = m * self.agents[i][0].args.gamma + (1 - m) * agent_j[0].args.gamma
        # gae_lambda
        self.agents[i][0].args.gae_lambda = m * self.agents[i][0].args.gae_lambda + (1 - m) * agent_j[0].gae_lambda

    def mutate(self, i, p_mutate=0.1, p_perturb=0.2):
        hyperparameters = [self.agents[i][0].args.learning_rate, self.agents[i][0].args.gamma,
                           self.agents[i][0].args.gae_lambda]    # , self.agents[i][0].arg.
        perturb = (1 + np.random.uniform(-p_perturb, p_perturb))
        for j in range(len(hyperparameters)):
            if np.random.random() < p_mutate:
                hyperparameters[j] = hyperparameters[j] * perturb
                if hyperparameters[j] > 1:
                    hyperparameters[j] = 1
                elif hyperparameters[j] < 0:
                    hyperparameters[j] = 0
        self.agents[i][0].args.learning_rate, self.agents[i][0].args.gamma, self.agents[i][0].args.gae_lambda = hyperparameters
        self.agents[i][0].optimizer.learning_rate = self.agents[i][0].args.learning_rate

    def teams(self, number_agents=16):
        self.agents = sorted(self.agents, key=itemgetter(1))
        indexes = []
        for i in reversed(range(number_agents)):
            indexes.append(self.agents[i][5])
        teams = []
        names = []
        count = 0
        for j in range(int(number_agents/2)):
            teams.append([indexes[j], indexes[-1-count]])
            names.append("Team_{}_{}".format(indexes[j], indexes[-1-count]))
            count += 1
        return teams, names

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret


class Experiment:
    def __init__(self, env, max_steps=1500):
        self.env = env
        self.max_steps = max_steps

    def training_fight(self, agents, render=False):
        # Agents = [self.agents[i], self.agents[j], self.agents[k], self.agents[l]]
        # where : team_1 = [i, j] and team_2 = [k, l]

        total_scores = [0 for _ in range(2)]            # 2 teams

        observation = obs_process(self.env.reset())
        step = 0
        while step < self.max_steps:
            action = tuple([agents[i][0].pre_step_train(step, observation[i]) for i in range(len(agents))])
            next_observation, reward, done, infos = self.env.step(action)
            next_observation = obs_process(next_observation)
            for i in range(len(agents)):
                agents[i][0].post_step(step, next_observation[i], reward[i], done)
            if render:
                self.env.render()
            # Add frame for learning since the beginning of training
            agents[:][2] += [1, 1, 1, 1]
            for i in range(len(agents)):
                # Add frame for learning since the last time it became eligible for evolution
                if agents[i][3] is not None:
                    agents[i][3] += 1
                # Add frame for learning since it last evolved
                if agents[i][4] is not None:
                    agents[i][4] += 1
            step += 1
            observation = next_observation
            if done:
                self.update(agents)
                break

        return total_scores

    @staticmethod
    def update(agents):      # Update params
        for n in range(len(agents)):
            agents[n][0].update()

    def evaluation_fight(self, agents, episode):
        # Agents = [self.agents[i], self.agents[j], self.agents[k], self.agents[l]]
        # where : team_1 = [i, j] and team_2 = [k, l]

        total_scores = [0 for _ in range(2)]  # 2 teams

        observation = obs_process(self.env.reset())
        step = 0
        while step < self.max_steps:
            action = tuple([agents[i].pre_step_eval(observation[i]) for i in range(len(agents))])
            next_observation, reward, done, infos = self.env.step(action)
            next_observation = obs_process(next_observation)
            self.env.render()
            step += 1
            observation = next_observation
            if done:
                if infos['termination'] == 'timeup':  # Time-up or every agent is out
                    print("{}. Game Tied   -   Scores: {} \t Total Steps: {}".format(episode, total_scores, step))
                    break
                if infos['status'][1] == 'defeated':  # Team 1 wins
                    total_scores[0] += 1
                    print("{}. Winner : Team 1 \t Scores: {} \t Total Steps: {}".format(episode, total_scores[0], step))
                    break
                elif infos['status'][0] == 'defeated':  # Team 2 wins
                    total_scores[1] += 1
                    print("{}. Winner : Team 2 \t Scores: {} \t Total Steps: {}".format(episode, total_scores[1], step))
                    break

        return total_scores
