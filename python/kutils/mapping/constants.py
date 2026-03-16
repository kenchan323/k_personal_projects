

B_IN_A_YEAR = 250
D_IN_A_YEAR = 365
W_IN_A_YEAR = 52
M_IN_A_YEAR = 12
Y_IN_A_YEAR = 1

def annualisation_factor(freqstr):
    if freqstr == 'B':
        return B_IN_A_YEAR
    if freqstr == 'D':
        return D_IN_A_YEAR
    if freqstr == 'BM':
        return M_IN_A_YEAR
    if 'W' in freqstr:
        return W_IN_A_YEAR
    if freqstr.startswith('A-'): # e.e.g 'A-DEC'
        return Y_IN_A_YEAR
    else:
        raise ValueError(f'{freqstr=} not recognised!')