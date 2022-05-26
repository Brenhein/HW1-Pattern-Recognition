import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def plot_histograms(data):
    """This function plots the histograms for p(x1|w) and p(x2|w) for each 
    class"""
    data_w = {}
    for i in range(2):
        for c, vals in data.items():
            data_w["\u03C9" + str(c)] = [v[i] for v in vals]
            
        # Plots the histograms
        df = pd.DataFrame(data_w)
        mini = math.floor(min(df["\u03C91"].min(), df["\u03C92"].min()))
        maxi = math.ceil(max(df["\u03C91"].max(), df["\u03C92"].max()))
        plot = df.plot.hist(bins=(maxi-mini), alpha=.5,
                            title="Feature " + str(i + 1), range=(mini, maxi))
        plot.set_xlabel("x" + str(i + 1))
        data_w = {}
        

def error_rate(data, con):
    """This function finds the percentage of misclassified datapoints"""
    length, incorrect = 0, 0
    for c, vals in data.items():
        length += len(vals)
        for v in vals:
            # Where doe the point fall??
            res = v[0] + v[1] - con
            what_class = 1 if res < 0 else 2
            
            # Is the result correct
            if what_class != c:
                incorrect += 1

    return incorrect / length          

def plot_scatter(data, con):
    """This function plots a scatter plot of the 2 data points for each class"""
    fig, ax = plt.subplots()
    
    # Plots the 2 points
    for c, vals in data.items():
        ax.scatter(*zip(*vals), label="\u03C9" + str(c))
        
    # Plot the decision boundary
    x = np.arange(-4, 16)
    ax.plot(x, con-x)
    
    # Plot nicities
    ax.legend()
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Class 1 vs Class 2")
    
    print("Error Rate for Decision Boundary:", error_rate(data, con))
        

def main():
    file = open("data.txt", "r")
    data = {1: [], 2: []}
    
    # Parses the file
    for line in file:
        line = line.strip().split()
        data[int(line[2])].append([float(line[0]), float(line[1])])
    
    plot_histograms(data)
    plot_scatter(data, 15)
    plot_scatter(data, 12)


if __name__ == "__main__":
    main()