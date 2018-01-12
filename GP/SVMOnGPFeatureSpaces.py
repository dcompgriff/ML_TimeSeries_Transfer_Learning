import numpy as np
import re
from sklearn import svm
from sklearn.model_selection import train_test_split
import argparse

description='''
This script takes as input a file generated using 'BuildDifferentFeatureSpace.py'\n
It runs SVM on the feature spaces in that file and produces avg error rate.\n
You can choose which kernel to test or all of them
'''
helpSourceData='''
Input .csv training file that has multiple feature representation af a set of labeled\n
data series, generated using 'BuildDifferentFeatureSpace.py'\n
'''
helpTask='''
Choose the task to classify as one vs all. Default='walk'\n
'''
helpKernel='''
Choose a kernel name to get error on a particular feature representation.\n
Default, enumerates over all feature representations.
'''
helpPercent='''
choice of test size as fraction in test train split. Default = 0.25
'''
p=re.compile("\s+")
def errorInSVM(X,y,test_percent=0.25):
    '''
    Runs SVM using polynomial kernel on the data, split between test and train.
    Returns the number of mistakes made on the test data and the size of test data.
    '''
    clf = svm.SVC()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_percent)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    errors=np.sum(y_pred != y_test)
    return errors, y_test.shape[0]
def getError(X,y,split,task):
    X=[[float(s) for s in xi[1:-2].split(",")] for xi in X]
    y=[1 if yi==task else -1 for yi in y]
    return errorInSVM(np.array(X),np.array(y),split)

def main(args):
    src=open(args.sourceData)
    split=float(args.split)
    task=args.task
    X=[]
    y=[]
    t=[]
    kernel=''
    if not bool(args.kernel):
        for line in src:
            if 'Kernel:' in line:
                if(len(y)!=0):
                    print(kernel, getError(X,y,split,task))
                kernel=line[8:]
                X=[]
                y=[]
            else:
                t=line.split('\t')
                X.append((t[1]))
                y.append(t[2][:-1])
        print(kernel, getError(X,y,split,task))
    else:
        for line in src:
            if kernel!=args.kernel:
                continue
            if 'Kernel:' in line:
                kernel=line[8:]
            else:
                t=line.split('\t')
                X.append((t[1]))
                y.append(t[2][:-1])
        print(kernel, getError(X,y,split,task))                
    src.close()    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('sourceData', help=helpSourceData)
    parser.add_argument('-task', default='sit', help=helpTask)
    parser.add_argument('-kernel', default='', help=helpKernel)
    parser.add_argument('-split', default=0.25, help=helpPercent)
    args = parser.parse_args()

    main(args)