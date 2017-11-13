import numpy as np
import pandas as pd
import cvxpy as cvx

# # Load all phone accelerometer data.
# phoneAccelData = pd.read_csv("../Activity recognition exp/Phones_accelerometer.csv")
# # Load a single set of time for data.
# dat1Entry = phoneAccelData[0:1362520]



def main():
    print("Inductive SVM Code.")

    Xs = np.ones((20,20))
    ys = np.ones((20, 1))
    Xt = np.ones((20, 20))
    yt = np.ones((20, 1))

    lambda1 = 1
    lambda2 = 1
    nt = 20  # The number of target examples.
    ns = 20  # The number of source examples.
    sd = 20  # The number of unshared dimensions in the source domain
    td = 20  # The number of unshared dimensions in the target domain
    wd = 20  # The number of shared dimensions between source and target domain

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
        constraints.append((yt[i] * Xt[i, :] * (w0 + vt) >= 1 - epsilonT[i]))
    for i in range(0, ns):
        constraints.append((epsilonS[i] >= 0))
    for i in range(0, nt):
        constraints.append((epsilonT[i] >= 0))

    prob = cvx.Problem(loss, constraints)
    print("optimal value: %f"% prob.solve())
    print("Optimal vs: " + str(vs.value))
    print("Optimal vt: " + str(vt.value))
    print("Optimal w0: " + str(w0.value))



if __name__ == "__main__":
    main()









