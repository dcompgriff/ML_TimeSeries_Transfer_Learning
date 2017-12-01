#imports
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import sklearn.gaussian_process.kernels as kers;

#Constants
myKernels={
    'k_Constant':kers.ConstantKernel(),
    'k_WN':kers.WhiteKernel(),
    'k_RBF':kers.RBF(),
    'k_RQ':kers.RationalQuadratic(),
    'k_mat0':kers.Matern(nu=0.5),
    'k_mat1':kers.Matern(nu=1.5),
    'k_mat2':kers.Matern(nu=2.5),
    'k_sine':kers.ExpSineSquared(),
    'k_dot':kers.DotProduct(),
    #now combination kernels
    'k1':kers.Sum(kers.ConstantKernel(),kers.ExpSineSquared()),
    'k2':kers.Product(kers.ConstantKernel(),kers.ExpSineSquared()),
    'k3':kers.Sum(kers.ConstantKernel(),kers.Product(kers.ConstantKernel(),kers.ExpSineSquared())),
    'k4':kers.Sum(kers.ConstantKernel(),kers.Product(kers.RBF(),kers.ExpSineSquared())),
    'k5':kers.Sum(kers.Product(kers.ConstantKernel(),kers.RBF()),kers.Product(kers.ConstantKernel(),kers.ExpSineSquared())),
    'k6':kers.Product(kers.ConstantKernel(),kers.Product(kers.RBF(),kers.ExpSineSquared())),
    'k7':kers.Sum(kers.ConstantKernel(),kers.Product(kers.ConstantKernel(),kers.Product(kers.RBF(),kers.ExpSineSquared())))
    }
#for key, value in myKernels.items():
#    print(key+","+str(len(value.theta)))
#print(kers.Matern().get_params())
l=[[key,value] for key, value in myKernels.items()]
l=[[k[0],[p for p in k[1].hyperparameters]]for k in l]
#print([[key,[p['name'] for p in value.hyperparameters]] for key, value in myKernels.items()])
print(l)
