from os import system
import numpy as np
import matplotlib.pyplot as plt
import os

exitflag = False

fig, axs = plt.subplots (2)

while True:
    # open log file
    logFile = open (os.getcwd () + "/bin/Release/devLogFile.csv", "r")
    data = logFile.read ().splitlines ()

    # close log file
    logFile.close ()
    
    steps = []
    accuracy = []
    cost = []

    # parse log file
    for s in data:
        tokens = s.split (',')
        steps.append (float (tokens [0]))
        accuracy.append (float (tokens [1]))
        cost.append (float (tokens [2]))

    cost_norm = np.array (cost)
    cost_norm = cost_norm [~np.isnan (cost)]

    accuracy_norm = np.array (accuracy)
    accuracy_norm = accuracy_norm [~np.isnan (cost)]

    n = np.array (steps)
    n = n [~np.isnan (cost)]

    # graph log data
    axs [0].plot (n, cost_norm, '-b')
    axs [1].plot (n, accuracy_norm, '-r')
    plt.pause (0.5)