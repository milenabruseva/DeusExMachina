import sys
from datetime import datetime
import main

generations = 10
training_rounds = 1000
evaluation_rounds = 10
agents = ["DeusExMachinaA", "DeusExMachinaB", "DeusExMachinaC", "DeusExMachinaD"]
to_train = "4"
scenario = 'classic'

start_time = datetime.now()
print(start_time)
for i in range(generations):
    if training_rounds != 0:
        main.main(['play', '--agents', agents[0], agents[1],
                   agents[2], agents[3], '--train', to_train, '--scenario', scenario, '--n-rounds',
                   str(training_rounds),
                   '--no-gui'])
    main.main(['play', '--save-stats', '--agents', agents[0], agents[1],
                   agents[2], agents[3], '--scenario', scenario, '--n-rounds', str(evaluation_rounds),
               '--no-gui'])
    print("Generation: " + str(i))

