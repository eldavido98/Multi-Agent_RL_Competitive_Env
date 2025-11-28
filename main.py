import argparse
from rl import *
from env.masurvival_env import MaSurvival


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MaSurvival()


def evaluate():
    environment = PBT_MARL(env)
    exp = Experiment(env)
    environment.load()
    agents = environment.agents

    # Create teams
    teams, names = environment.teams()

    scores = []
    for f in range(len(teams) - 1):
        i, j = teams[f]
        for t in range(f + 1, len(teams)):
            if t != f:
                k, l = teams[t]
                s_i_j, s_k_l = exp.evaluation_fight([agents[i][0], agents[j][0], agents[k][0], agents[l][0]], f)
                scores.append(["Team_{}_{}".format(i, j), s_i_j])
                scores.append(["Team_{}_{}".format(k, l), s_k_l])
    freq = []
    for n in names:
        count = 0
        for i in range(len(scores)):
            if n == scores[i][0] and scores[i][1] == 1:
                count += 1
        freq.append(count)

    print("Matches won for each team:")
    for i in range(len(names)):
        print(names[i], freq[i])


def train():
    agents = PBT_MARL(env)
    agents.train(max_episodes=500)
    agents.save()


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()

    if args.evaluate:
        evaluate()


if __name__ == '__main__':
    main()
