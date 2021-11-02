from os import system
import numpy as np
import matplotlib.pyplot as plt
import os

exitflag = False

fig, axs = plt.subplots (2, 2)
axs [0, 0].set_title ("Batch Cost vs Step")
axs [1, 0].set_title ("Batch Accuracy vs Step")
axs [0, 1].set_title ("Epoch Cost vs Step")
axs [1, 1].set_title ("Epoch Accuracy vs Step")

while plt.fignum_exists (1):

    # open log file
    logFile = open (os.getcwd () + "/TrainingLog.csv", "r")
    data = logFile.read ().splitlines ()

    # close log file
    logFile.close ()
    
    steps = []
    batch_accuracy = []
    batch_cost = []
    epoch_accuracy = []
    epoch_cost = []

    # parse log file
    for s in data:
        tokens = s.split (',')
        steps.append (float (tokens [0]))
        batch_accuracy.append (float (tokens [1]))
        batch_cost.append (float (tokens [2]))
        epoch_accuracy.append (float (tokens [3]))
        epoch_cost.append (float (tokens [4]))


    n = np.array (steps)
    n = n [~np.isnan (batch_cost)]

    # graph log data
    axs [0, 0].plot (n, batch_cost, '-b')
    axs [1, 0].plot (n, batch_accuracy, '-r')
    axs [0, 1].plot (n, epoch_cost, '-b')
    axs [1, 1].plot (n, epoch_accuracy, '-r')
    plt.pause (1)