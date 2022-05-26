import pprint
import scipy.spatial as sc
import matplotlib.pyplot as plt
import pandas as pd

DIMENSIONS = 8
PATTERNS = 48
TITLES = ["Upper Left Boundary", "Lower Right Boundary", \
          "Upper Right Boundary", "Lower Left Boundary", \
          "Middle Left Boundary", "Middle Right Boundary", \
          "Middle Upper Boundary", "Middle Lower Boundary"]
NAME_MAP = {1: "I", 2: "M", 3: "O", 4: "X"}

    
def mean_pattern_vector(data):
    """Computes the sum of the data points"""
    d_avgs = {}
    print("\nMean Pattern Vectors:\n")
    for c, vals in data.items():
        d_sum = [0] * DIMENSIONS
        for d in vals:
            for i in range(DIMENSIONS):
                d_sum[i] += d[i]
                
        # Computes the average
        d_avg = [round(el / PATTERNS, 3) for el in d_sum]
        d_avgs[c] = d_avg
        print("Class " + str(c) + ":", d_avg)
    return d_avgs
    

def draw_scatter_3d(x, y, z, cols, data):
    """This function draws a 3d scatter plot"""
    patterns = []
    for c, vals in data.items():
        for v in vals:
            patterns.append([NAME_MAP[c], v[x], v[y], v[z]])
    
    df = pd.DataFrame(patterns, columns=cols)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(list(df[cols[1]])[0:48], list(df[cols[2]])[0:48],
               list(df[cols[3]])[0:48], c='blue', label="I", alpha=.4)
    ax.scatter(list(df[cols[1]])[48:96], list(df[cols[2]])[48:96], 
               list(df[cols[3]])[48:96], c='orange', label="M", alpha=.4)
    ax.scatter(list(df[cols[1]])[96:144], list(df[cols[2]])[96:144],
               list(df[cols[3]])[96:144], c='green', label="O", alpha=.4)
    ax.scatter(list(df[cols[1]])[144:192], list(df[cols[2]])[144:192], 
               list(df[cols[3]])[144:192], c='red', label="X", alpha=.4)
    
    ax.set_xlabel(cols[1] + " Distance")
    ax.set_ylabel(cols[2] + " Distance")
    ax.set_zlabel(cols[3] + " Distance")
    ax.set_title("Boundary Distances")
    ax.legend()  


def draw_scatter(x, y, cols, data):
    """This function draws a scatter plot for 2 features, grouped by class"""
    patterns = []
    for c, vals in data.items():
        for v in vals:
            patterns.append([NAME_MAP[c], v[x], v[y]])
    
    df = pd.DataFrame(patterns, columns=cols)
    
    fig, ax = plt.subplots()
    ax.scatter(x=list(df[cols[1]])[0:48], y=list(df[cols[2]])[0:48], c='blue',
               label="I", alpha=.4)
    ax.scatter(x=list(df[cols[1]])[48:96], y=list(df[cols[2]])[48:96], 
               c='orange', label="M", alpha=.4)
    ax.scatter(x=list(df[cols[1]])[96:144], y=list(df[cols[2]])[96:144],
               c='green', label="O", alpha=.4)
    ax.scatter(x=list(df[cols[1]])[144:192], y=list(df[cols[2]])[144:192], 
               c='red', label="X", alpha=.4)
    
    ax.set_xlabel(cols[1] + " Distance")
    ax.set_ylabel(cols[2] + " Distance")
    ax.set_title("Boundary Distances")
    ax.legend()
    
    
def euclidean_distance(data, d_avgs):
    """Computes the Euclidean Distance Formula"""
    print("\nFarthest Patterns From The Mean:\n")
    for c, vals in data.items():
        max_dist, max_num = 0, -1
        for i, d in enumerate(vals):
            dist = sc.distance.euclidean(d, d_avgs[c])
            if dist > max_dist:
                print(dist, i)
                max_dist = dist
                max_num = i
        print("Class " + str(c) + ":", vals[max_num], \
              "(Distance = " + str(round(max_dist, 3)) +")")        
    
 
def main():
    # Initializes the data
    file = open("imox_data.txt", "r")
    data = {1: [], 2: [], 3: [], 4: []}
    d_avgs = {}
    
    # Parses the file
    for line in file:
        line = line.strip().split()
        line = [int(float(d)) for d in line]
        data[line[-1]].append(line[:-1])
    
    d_avgs = mean_pattern_vector(data)
    euclidean_distance(data, d_avgs)
            
    # Converts the data into pandas dataframes 
    for i in range(DIMENSIONS):
        feature_cols = {"I": [], "M": [], "O": [], "X": []}
        
        # Gets the features
        for c, vals in data.items():
            for v in vals:
                feature_cols[NAME_MAP[c]].append(v[i])
                
        # Plots the histograms for each of the 8 features
        df = pd.DataFrame(feature_cols)
        plot = df.plot.hist(bins=12, alpha=.5, title=TITLES[i])
        plot.set_xlabel("Distance")
            
    draw_scatter(0, 1, ["Letter", "Upper Left", "Lower Right"], data)
    draw_scatter(2, 3, ["Letter", "Upper Right", "Lower Left"], data)
    draw_scatter_3d(0, 1, 3, ["Letter", "Upper Left", "Lower Right", \
                              "Lower Left"], data)
    
    
if __name__ == "__main__":
    main()
    
    
        
    
