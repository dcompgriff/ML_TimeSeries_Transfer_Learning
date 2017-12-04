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