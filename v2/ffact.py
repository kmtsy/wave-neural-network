import itertools
import numpy as np

# generates full factorial table 
def generate_factorial_table(shlb, shub, shs, phlb, phub, phs, irlb, irub, irs, pvlb, pvub, pvs):
    
    # create variable ranges 
    var1_range = np.arange(shlb,shub + shs,shs)
    var2_range = np.arange(phlb,phub + phs,phs)
    var3_range = np.arange(irlb,irub + irs,irs)
    var4_range = np.arange(pvlb,pvub + pvs,pvs)

    # generate combinations of all variables
    combinations = list(itertools.product(var1_range, var2_range, var3_range, var4_range))
    
    return combinations