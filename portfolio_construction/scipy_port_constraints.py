import numpy as np

'''
Defining the constraints used in scipy based optimisation
'''

def total_weight_constraint(x):
    '''
    Total weight constraint to be used by scipy solver
    :param x:
    :return:
    '''
    return np.sum(x)-1.0

def long_only_constraint(x):
    '''
    Long only constraint to be used by scipy solver to ensure w > 0
    :param x:
    :return:
    '''
    return x