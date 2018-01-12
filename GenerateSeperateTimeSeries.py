import numpy as np
import pandas as pd
import argparse
description='''Given a input CSV file containing data from multiple users and multiple devices, 
this routine will break the data into separate files each containing samples just for a single 
sensor and user.'''
def fileName(user, isPhone, isGyro, device):
    s="Phone" if isPhone else "Watch"
    s+= "-Gyro" if isGyro else "-Acc"
    s+="-"+device+"-"+user+".csv"
    return s
def main(args):
    cols=["Index","Arrival_Time","Creation_Time","x","y","z","User","Model","Device","gt"]
    data=pd.read_csv(args.input_file)
    isPhone="Phone" in args.input_file
    isGyro="gyro" in args.input_file
    #Output rows may not match input rows due to floating point precision errors
    users=data["User"].unique()
    devices=data["Device"].unique()
    #device name includes model name
    #models=data["Model"].unique() 
    for user in users:
        for device in devices:
            #for model in models:
            d=data.loc[(data['User']==user) & (data['Device']==device)]
            d.to_csv(fileName(user, isPhone, isGyro, device),index=False)
    #data.to_csv(fileName('a', False, True, 'd', 'm'),index=False )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input_file', help='Input .csv file to read sensor data from.')
    args = parser.parse_args()
    
    main(args)