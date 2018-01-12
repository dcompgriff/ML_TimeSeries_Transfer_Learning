import sklearn.gaussian_process.kernels as kers;
#Constants
myKernels={
    'k_Constant': lambda : kers.ConstantKernel(),
    'k_WN': lambda : kers.WhiteKernel(),
    'k_RBF': lambda : kers.RBF(),
    'k_RQ': lambda : kers.RationalQuadratic(),
    'k_mat0': lambda : kers.Matern(nu=0.5),
    'k_mat1': lambda : kers.Matern(nu=1.5),
    'k_mat2': lambda : kers.Matern(nu=2.5),
    'k_sine': lambda : kers.ExpSineSquared(),
    #now combination kernels
    'k1': lambda : kers.Sum(kers.ConstantKernel(),kers.ExpSineSquared()),
    'k2': lambda : kers.Product(kers.ConstantKernel(),kers.ExpSineSquared()),
    'k3': lambda : kers.Sum(kers.ConstantKernel(),kers.Product(kers.ConstantKernel(),kers.ExpSineSquared())),
    'k4': lambda : kers.Sum(kers.ConstantKernel(),kers.Product(kers.RBF(),kers.ExpSineSquared())),
    'k5': lambda : kers.Sum(kers.Product(kers.ConstantKernel(),kers.RBF()),kers.Product(kers.ConstantKernel(),kers.ExpSineSquared())),
    'k6': lambda : kers.Product(kers.ConstantKernel(),kers.Product(kers.RBF(),kers.ExpSineSquared())),
    'k7': lambda : kers.Sum(kers.ConstantKernel(),kers.Product(kers.ConstantKernel(),kers.Product(kers.RBF(),kers.ExpSineSquared())))
    }