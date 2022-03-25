import json
from datetime import datetime
import matplotlib.pyplot as plt
import main
import glob, os

generations = 10
training_rounds = 100
evaluation_rounds = 10
agent = 'q-learning'
scenario = 'classic'
opponents = ['rule_based_agent', 'rule_based_agent', 'rule_based_agent']
#opponents = ['peaceful_agent', 'peaceful_agent', 'peaceful_agent']
date_format = "%Y-%m-%d %H-%M-%S"

x = range(generations)
metrics = ["invalid", "score", "time", "suicides", "bombs", "crates", "kills", "moves", "diff"]
y = {k:[] for k in metrics}

start_time = datetime.now()
for i in range(generations):
    if training_rounds != 0:
        main.main(['play', '--agents', agent, opponents[0],
                opponents[1], opponents[2], '--train', '1', '--scenario', scenario, '--n-rounds', str(training_rounds),
                '--no-gui'])
    main.main(['play', '--save-stats', '--agents', agent, opponents[0],
                opponents[1], opponents[2], '--scenario', scenario, '--n-rounds', str(evaluation_rounds),
                '--no-gui'])
    print("Generation: " + str(i))

os.chdir("./results")
for file in glob.glob("*.json"):
    datetime_str = datetime.strptime(file.split('.')[0], date_format)
    if start_time < datetime_str:
        results = json.load(open(file, 'r'))
        results_agent = results["by_agent"][agent]

        for metric in metrics:
            if metric in results_agent and metric != "diff":
                y[metric].append(results_agent[metric])
            elif metric == "diff":
                score_agent = results_agent["score"]
                scores_opponents = []
                for opponent_i in range(len(opponents)):
                    scores_opponents.append(results["by_agent"][opponents[opponent_i] + "_" + str(opponent_i)]['score'])
                y["diff"].append(score_agent - max(scores_opponents))
            else:
                y[metric].append(0.0)


os.chdir("..")
path = os.path.join(os.getcwd(), "graphs", datetime.now().strftime("%Y%m%d%H%M%S"))
os.mkdir(path)

for metric in metrics:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x, y[metric])
    fig.savefig(os.path.join(path, metric + '.png'))


