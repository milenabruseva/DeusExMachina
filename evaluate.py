from datetime import datetime
import main
from plot_stats import plot_stats

generations = 10
training_rounds = 0
evaluation_rounds = 1
agent = 'q-learning'
scenario = 'classic'
opponents = ['rule_based_agent', 'rule_based_agent', 'rule_based_agent']

start_time = datetime.now()
print(start_time)
for i in range(generations):
    if training_rounds != 0:
        main.main(['play', '--agents', agent, opponents[0],
                   opponents[1], opponents[2], '--train', '1', '--scenario', scenario, '--n-rounds',
                   str(training_rounds),
                   '--no-gui'])
    main.main(['play', '--save-stats', '--agents', agent, opponents[0],
               opponents[1], opponents[2], '--scenario', scenario, '--n-rounds', str(evaluation_rounds),
               '--no-gui'])
    print("Generation: " + str(i))

plot_stats(start_time)
