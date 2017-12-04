#imports
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import sklearn.gaussian_process.kernels as kers;
from myKernels import myKernels
from sklearn.gaussian_process import GaussianProcessRegressor
import argparse
import warnings
warnings.filterwarnings("ignore")

description='''
This script converts given time series to multiple feature representations.\n
Provide a training file, after generating from GenerateTrainingExamples.py.\n
Prints out to a file, named by suffixing original file name with '_features'.\n
Writes different feature space representations of the given time series under different kernels.\n
'''
helpSourceData='''
Input .csv training file that contains fixed width raw time series data,\n
generated using GenerateTrainingExamples.py
'''
helpT='''
The fixed window size used for the file. Default=500
'''
helpTask='''
T
'''

#for key, value in myKernels.items():
#    print(key+","+str(len(value.theta)))
#print(kers.Matern().get_params())
#l=[[key,value] for key, value in myKernels.items()]
#l=[[k[0],[p for p in k[1].hyperparameters]]for k in l]
#print([[key,[p['name'] for p in value.hyperparameters]] for key, value in myKernels.items()])
#print(l)



def makeTimeSeries(data,t):
    '''
    Makes time series of (t,x,y,z) labeled by {1,-1} for the task.
    Returns a tuple of arrays, indexed for each of these time series. 
    '''
    previousClassLabel = str(data.at[0, 'gt'])
    pos = 0
    window = t
    t = []
    xt = []
    yt = []
    zt = []
    l = []
    while pos < data.shape[0]:
        # Make l label.
        l.append(data.iloc[pos]['gt'])
        # Make X row.
        xt.append(data.iloc[pos:pos + window]['x'])
        yt.append(data.iloc[pos:pos + window]['y'])
        zt.append(data.iloc[pos:pos + window]['z'])
        t.append(data.iloc[pos:pos + window]['Arrival_Time'])
        # Move to the next window
        pos += window
    return np.array(t), np.array(xt), np.array(yt), np.array(zt), np.array(l)

def getGaussianFeatureSpace(kernel, t, x, y, z, left = 245, right = 255):
    '''
    Given a kernel and a time series point(t,x,y,z), fits each series of x,y,z to a GP model
    with the kernel. Returns a list of features in this new feature space. 
    '''
    finalFeatures = []
    for i in range(left,right):
        # kernel = RationalQuadratic()
        print("Done: ", i)
        finalFeatures.append([])
        gpx = GaussianProcessRegressor(kernel,optimizer='fmin_l_bfgs_b',\
                                       copy_X_train=False, normalize_y=True)
        gpx.fit(t[i,:].reshape(-1, 1) - t[i,0], x[i,:])
        
        gpy = GaussianProcessRegressor(kernel,optimizer='fmin_l_bfgs_b',\
                                       copy_X_train=False, normalize_y=True)
        gpy.fit(t[i,:].reshape(-1, 1) - t[i,0], y[i,:])
        
        gpz = GaussianProcessRegressor(kernel,optimizer='fmin_l_bfgs_b',\
                                       copy_X_train=False, normalize_y=True)
        gpz.fit(t[i,:].reshape(-1, 1) - t[i,0], z[i,:])
        
        px = gpx.kernel_
        py = gpy.kernel_
        pz = gpz.kernel_
        [finalFeatures[-1].append(x_) for x_ in px.theta]
        [finalFeatures[-1].append(x_) for x_ in py.theta]
        [finalFeatures[-1].append(x_) for x_ in pz.theta]
    print("Done")
    return np.array(finalFeatures)


#A constant to the directory where training files are located. Point this to corresponding folder on your machine.  
#trainDataFolder="G:/edu/coding/Activity recognition exp/train/"
#Run GenerateTrainingExamples on a single source data file, to obtain fixed width time series
#sourceData = trainDataFolder+"Phone-Acc-nexus4_1-a_train.csv"
#targetData = trainDataFolder+"Phone-Acc-nexus4_1-b_train.csv"

#outFile=trainDataFolder+sourceData[:-4]+"_features.csv"
#classification task
#task='walk'
#Ds = pd.read_csv(sourceData)

#ts, xts, yts, zts, ls = makeTimeSeries(Ds)
#TODO run SVM error on raw sampled time series.

#outFile=sourceData[:-4]+"_features.csv"
#f = open(outFile,'w')
#for k_,kernel in myKernels.items():
#    f.write("Kernel: "+k_+"\n")
#    left = 267
#    right = 269
#    Xmod = getGaussianFeatureSpace(kernel, ts, xts, yts, zts, left, right)
#    ymod = ls[left:right]
#    for i in range(0,len(ymod)):
#        f.write(str(i)+"\t"+str(Xmod[i])+"\t"+str(ymod[i])+"\n")
#f.close()



def main(args):
    sourceData=args.sourceData
    width=args.t
    #read data
    Ds = pd.read_csv(sourceData)
    #break into data as time series
    ts, xts, yts, zts, ls = makeTimeSeries(Ds,width)
    outFile=sourceData[:-4]+"_features.csv"
    f = open(outFile,'w')
    for k_,kernel in myKernels.items():#for each kernel
        print(k_)
        f.write("Kernel: "+k_+"\n")
        left = 0
        right = len(ls)
        Xmod = getGaussianFeatureSpace(kernel, ts, xts, yts, zts, left, right)
        ymod = ls[left:right]
        for i in range(0,len(ymod)):
            f.write(str(i)+"\t"+np.array2string(Xmod[i],separator=",")+"\t"+str(ymod[i])+"\n")
        break
    f.close()

#errors,testSize=errorInSVM(Xmod,ymod)

#print("Incorrect classifications", errors , "->", 100*errors/testSize, "%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('sourceData', help=helpSourceData)
    parser.add_argument('-t', type=int, default=500, help=helpT)
    #parser.add_argument('-task', default='walk', help=helpTask)
    args = parser.parse_args()

    main(args)