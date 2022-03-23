import json
from datetime import datetime
import matplotlib.pyplot as plt
import main
import glob, os
import statistics

generations = 10
training_rounds = 100
evaluation_rounds = 1
agent = 'sarsa-lambda'
scenario = 'coin-heaven'
opponents = ['rule_based_agent', 'rule_based_agent', 'rule_based_agent']
date_format = "%Y-%m-%d %H-%M-%S"


x = range(generations)
y_invalid = []
y_score = []
y_time = []
y_suicides = []
y_bombs = []
y_crates = []
y_kills = []
y_diff = []

for i in range(generations):
    start_time = datetime.now()
    main.main(['play', '--agents', agent, opponents[0],
                opponents[1], opponents[2], '--train', '1', '--scenario', scenario, '--n-rounds', str(training_rounds),
                '--no-gui'])
    main.main(['play', '--save-stats', '--agents', agent, opponents[0],
                opponents[1], opponents[2], '--scenario', scenario, '--n-rounds', str(evaluation_rounds),
                '--no-gui'])
    os.chdir("./results")
    y_invalid_tmp = []
    y_score_tmp = []
    y_time_tmp = []
    y_suicides_tmp = []
    y_bombs_tmp = []
    y_crates_tmp = []
    y_kills_tmp = []
    y_diff_tmp = []
    for file in glob.glob("*.json"):
        datetime_str = datetime.strptime(file.split('.')[0], date_format)
        if start_time < datetime_str:
            results = json.load(open(file, 'r'))
            results_agent = results["by_agent"][agent]
            score_agent = results_agent["score"]

            y_score_tmp.append(score_agent)
            y_time_tmp.append(results_agent["time"])
            if "invalid" in results_agent:
                y_invalid_tmp.append(results_agent["invalid"])
            else:
                y_invalid_tmp.append(0.0)
            if "suicides" in results_agent:
                y_suicides_tmp.append(results_agent["suicides"])
            else:
                y_suicides_tmp.append(0.0)
            if "bombs" in results_agent:
                y_bombs_tmp.append(results_agent["bombs"])
            else:
                y_bombs_tmp.append(0.0)
            if "crates" in results_agent:
                y_crates_tmp.append(results_agent["crates"])
            else:
                y_crates_tmp.append(0.0)
            if "kills" in results_agent:
                y_kills_tmp.append(results_agent["kills"])
            else:
                y_kills_tmp.append(0.0)
            scores_opponents = []
            for opponent_i in range(len(opponents)):
                scores_opponents.append(results["by_agent"][opponents[opponent_i] + "_" + str(opponent_i)]['score'])
            y_diff_tmp.append(score_agent - max(scores_opponents))

    y_invalid.append(statistics.mean(y_invalid_tmp))
    y_score.append(statistics.mean(y_score_tmp))
    y_time.append(statistics.mean(y_time_tmp))
    y_suicides.append(statistics.mean(y_suicides_tmp))
    y_bombs.append(statistics.mean(y_bombs_tmp))
    y_crates.append(statistics.mean(y_crates_tmp))
    y_kills.append(statistics.mean(y_kills_tmp))
    y_diff.append(statistics.mean(y_diff_tmp))
    print("Generation: " + str(i) + " Diff: " + str(statistics.mean(y_diff_tmp)))
    os.chdir("..")


path = os.path.join(os.getcwd(), "graphs", datetime.now().strftime("%Y%m%d%H%M%S"))
os.mkdir(path)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, y_score)
fig.savefig(os.path.join(path, 'score.png'))
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, y_invalid)
fig.savefig(os.path.join(path, 'invalid.png'))
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, y_time)
fig.savefig(os.path.join(path, 'time.png'))
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, y_suicides)
fig.savefig(os.path.join(path, 'suicides.png'))
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, y_bombs)
fig.savefig(os.path.join(path, 'bombs.png'))
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, y_crates)
fig.savefig(os.path.join(path, 'crates.png'))
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, y_kills)
fig.savefig(os.path.join(path, 'kills.png'))
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, y_diff)
fig.savefig(os.path.join(path, 'diff.png'))

