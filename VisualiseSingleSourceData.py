import numpy as np
import pandas as pd;
import matplotlib.pyplot as plt
import argparse
description='''
This script generates a plot of data from a single user. The plot's x axis is the arrival
time stamps. The y axis is the 'x' reading of the sensor. Each activity in this data
is labelled using a different color so that we can visualize what kind of waveforms occur 
for different activities.
'''
colors={
    'stand': '#ff0000', 
    np.nan: '#000000', 
    'sit': '#00ff00', 
    'walk': '#0000ff', 
    'stairsup': '#ffff00',  
    'bike': '#00ffff', 
    'stairsdown': '#ff00ff'
    }
linewidth=0.2
def main(args):
    data=pd.read_csv(args.input_file)
    tasks=data['gt'].unique()
    plt.figure('Sensor readings, colored by activity')
    for t in tasks:
        dt=data.loc[data['gt']==t]
        time=dt['Arrival_Time']
        
        plt.subplot(311)
        plt.ylabel('X')
        line,=plt.plot(time,dt['x'],c=colors[t], label=t)
        plt.setp(line, linewidth=linewidth)
        
        plt.subplot(312)
        plt.ylabel('Y')
        line,=plt.plot(time,dt['y'],c=colors[t], label=t)
        plt.setp(line, linewidth=linewidth)
        
        plt.subplot(313)
        plt.ylabel('Z')
        line,=plt.plot(time,dt['z'],c=colors[t], label=t)
        plt.setp(line, linewidth=linewidth)
        
    plt.legend()
    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input_file', help='Input single source .csv file to read sensor data from.')
    args = parser.parse_args()
    
    main(args)