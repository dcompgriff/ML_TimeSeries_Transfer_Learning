import numpy as np
import pandas as pd
import cvxpy as cvx
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC

# # Load all phone accelerometer data.
# phoneAccelData = pd.read_csv("../Activity recognition exp/Phones_accelerometer.csv")
# # Load a single set of time for data.
# dat1Entry = phoneAccelData[0:1362520]

def calculateTotalAbsoluteError(yPredicted, yOriginal):
    count = 0
    for i in range(0, yOriginal.shape[0]):
        if yPredicted[i] != yOriginal[i]:
            count += 1
    return count

def predict(X, w):
    ypredicted = []
    for x in X:
        yhat = x.dot(w)
        if yhat >= 0:
            ypredicted.append(1)
        else:
            ypredicted.append(-1)

    return np.array(ypredicted).reshape((len(ypredicted), 1))

def irisISVM():
    print("Iris Inductive SVM Code.")

    data = load_iris()
    # Xs = np.ones((20,20))
    # ys = np.ones((20, 1))
    # Xt = np.ones((20, 20))
    # yt = np.ones((20, 1))
    randomTargetIndexSet = np.random.randint(0, 150, 50)
    randomSourceIndexSet = list(set(range(0, 150)).difference(set(randomTargetIndexSet)))

    #Target = predict 0's from data set. Source = Predict 1's from data set.
    Xt = data.data[randomTargetIndexSet]
    Xs = data.data[randomSourceIndexSet]
    yt = data.target[randomTargetIndexSet]
    yt = np.array(list(map(lambda item: 1 if item == 0 else -1, yt.tolist())))
    ys = data.target[randomSourceIndexSet]
    ys = np.array(list(map(lambda item: 1 if item == 1 else -1, ys.tolist())))

    lambda1 = 1
    lambda2 = 1
    nt = 50  # The number of target examples.
    ns = 100  # The number of source examples.
    sd = 4  # The number of unshared dimensions in the source domain
    td = 4  # The number of unshared dimensions in the target domain
    wd = 4  # The number of shared dimensions between source and target domain

    # Set up the inductive transfer svm optimization formulation.
    vs = cvx.Variable(td, 1)
    vt = cvx.Variable(sd, 1)
    w0 = cvx.Variable(wd, 1)
    epsilonS = cvx.Variable(ns, 1)
    epsilonT = cvx.Variable(nt, 1)

    # Build the full loss function.
    ls = cvx.sum_entries(epsilonS) + ((lambda1/2.0)*cvx.sum_squares(vs))
    lt = cvx.sum_entries(epsilonT) + ((lambda1/2.0)*cvx.sum_squares(vt))
    loss = cvx.Minimize(ls + lt + (lambda2*cvx.sum_squares(w0)))

    # Set up the constraints.
    constraints = []
    for i in range(0, ns):
        constraints.append(( ys[i]*Xs[i, :]*(w0 + vs) >= 1 - epsilonS[i]))
    for i in range(0, nt):
        constraints.append(( yt[i]*Xt[i, :]*(w0 + vt) >= 1 - epsilonT[i]))
    for i in range(0, ns):
        constraints.append((epsilonS[i] >= 0))
    for i in range(0, nt):
        constraints.append((epsilonT[i] >= 0))

    prob = cvx.Problem(loss, constraints)
    print("Optimal value: %f"% prob.solve())
    print("Optimal vs: " + str(vs.value))
    print("Optimal vt: " + str(vt.value))
    print("Optimal w0: " + str(w0.value))

    # Make SVM for source.
    clfs = LinearSVC(random_state=0)
    clfs.fit(Xs, ys)
    # Make SVM for targes.
    clft = LinearSVC(random_state=0)
    clft.fit(Xt, yt)

    # Calculate and compare errors of ISVM.
    sourcePredictedY = predict(Xs, w0.value + vs.value)
    targetPredictedY = predict(Xt, w0.value + vt.value)
    sourceError = calculateTotalAbsoluteError(sourcePredictedY, ys)/100.0
    targetError = calculateTotalAbsoluteError(targetPredictedY, yt)/50.0
    print("Source domain error: %f"%sourceError)
    print("Target domain error: %f"%targetError)

    #Calculate and compare errors of regular SVM.
    regSourcePredictedY = predict(Xs, clfs.coef_.reshape(len(clfs.coef_.ravel()), 1))
    regTargetPredictedY = predict(Xt, clft.coef_.reshape(len(clft.coef_.ravel()), 1))
    regSourceError = calculateTotalAbsoluteError(regSourcePredictedY, ys)/100.0
    regTargetError = calculateTotalAbsoluteError(regTargetPredictedY, yt)/50.0
    print("Regular SVM Source domain error: %f"%regSourceError)
    print("Regular SVM Target domain error: %f"%regTargetError)


'''
Function applies both a regular SVM and ISVM to the task of classifying the
phone accelerometer data set.
'''
def phoneAccelerometerISVM():
    pass


if __name__ == "__main__":
    irisISVM()









