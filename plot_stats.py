import json
from datetime import datetime
import matplotlib.pyplot as plt
import glob, os

date_format = "%Y-%m-%d %H-%M-%S"
metrics = ["invalid", "score", "time", "suicides", "bombs", "crates", "kills", "moves", "diff", "steps"]


def plot_stats(start_time):
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
    os.chdir("./results")
    files_to_plot = [file for file in glob.glob("*.json") if
                     start_time < datetime.strptime(file.split('.')[0], date_format)]
    if len(files_to_plot):
        x = range(len(files_to_plot))
        results = json.load(open(files_to_plot[0], 'r'))
        opponents_all = list(results["by_agent"].keys())
        agents_y = {k: {} for k in opponents_all}
        for agent in opponents_all:
            agents_y[agent] = {k: [] for k in metrics}
        for file in files_to_plot:
            results = json.load(open(file, 'r'))
            for agent in opponents_all:
                opponents = opponents_all[:]
                opponents.remove(agent)
                agent_result = results["by_agent"][agent]
                for metric in metrics:
                    if metric in agent_result and metric != "diff":
                        agents_y[agent][metric].append(agent_result[metric])
                    elif metric == "diff":
                        score_agent = agent_result["score"]
                        scores_opponents = []
                        for opponent in opponents:
                            scores_opponents.append(
                                results["by_agent"][opponent]['score'])
                        agents_y[agent]["diff"].append(score_agent - max(scores_opponents))
                    else:
                        agents_y[agent][metric].append(0.0)

        os.chdir("..")
        path = os.path.join(os.getcwd(), "graphs", datetime.now().strftime("%Y%m%d%H%M%S"))
        os.mkdir(path)

        for agent in opponents_all:
            fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10, 17))
            for i, ax in enumerate(fig.axes):
                ax.plot(x, agents_y[agent][metrics[i]])
                ax.set_title(metrics[i].capitalize())
            fig.suptitle("Statistics for " + agent, fontsize=16)
            fig.savefig(os.path.join(path, agent + '.png'))

        fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10, 20))
        for i, ax in enumerate(fig.axes):
            for agent in opponents_all:
                ax.plot(x, agents_y[agent][metrics[i]], label=agent)
                ax.set_title(metrics[i].capitalize())
                ax.legend()
            fig.suptitle("Statistics for all agents", fontsize=16)
            fig.savefig(os.path.join(path, 'all_agents.png'))


def plot_learnability(agents, algorithm):
    f = open("./agent_code/" + agents[0] + "/rewards.txt", "r")
    agentA = [float(x.strip()) for x in f.readlines()]

    f = open("./agent_code/" + agents[1] + "/rewards.txt", "r")
    agentB = [float(x.strip()) for x in f.readlines()]

    f = open("./agent_code/" + agents[2] + "/rewards.txt", "r")
    agentC = [float(x.strip()) for x in f.readlines()]

    f = open("./agent_code/" + agents[3] + "/rewards.txt", "r")
    agentD = [float(x.strip()) for x in f.readlines()]

    x = range(len(agentA))
    path = os.path.join(os.getcwd(), "graphs", datetime.now().strftime("%Y%m%d%H%M%S"))
    os.mkdir(path)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax.plot(x, agentA, label=agents[0])
    ax.plot(x, agentB, label=agents[1])
    ax.plot(x, agentC, label=agents[2])
    ax.plot(x, agentD, label=agents[3])
    ax.set(xlabel="Number of training rounds times 100", ylabel="Average reward value per round")
    ax.legend()
    fig.suptitle("Learnability: " + algorithm, fontsize=16)
    fig.savefig(os.path.join(path, 'learnability_' + algorithm + '.png'))


plot_stats('2022-04-04 17:34:11.575268')
#agents = ["sarsa-pureA", "sarsa-pureB", "sarsa-pureC", "sarsa-pureD"]
#algorithm = "sarsa-pure"
# agents = ["sarsa-lambdaA", "sarsa-lambdaB", "sarsa-lambdaC", "sarsa-lambdaD"]
# algorithm = "sarsa-lambda"
# agents = ["q-learningA", "q-learningB", "q-learningC", "q-learningD"]
# algorithm = "q-learning"
#plot_learnability(agents, algorithm)
