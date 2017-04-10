import numpy as np
import parameters as par

def test_import():
    try:
        import trappist_1
    except:
        raise AssertionError#, "trappist_1.py import failed"

import trappist_1 as trp

def test_arrays():
    try:
        T = np.load('npy/T.npy')
        R = np.load('npy/R.npy')
        V = np.load('npy/V.npy')
    except:
        raise AssertionError#, "failed to load T, R, and/or V"

T = np.load('npy/T.npy')
R = np.load('npy/R.npy')
V = np.load('npy/V.npy')

def test_momentum():
    P = np.load('npy/P.npy')
    np.testing.assert_array_almost_equal(P,np.zeros_like(P), decimal=3)

def test_period():

    periods = []
    for i_b in trp.bodies[1:]:
        per = 0
        i_t   = 10
        while per == 0:
            if all(( R[i_b,i_t-1,1] < 0 , R[i_b,i_t,1] >= 0 )):
                per += 1
                periods.append( T[i_t] )
            else:
                i_t += 1

    calculated = np.array(periods)
    given = np.array(par.planets['period'])
    test = np.abs(calculated - given)/given
    print('calculated', calculated)
    print('given', given)
    print('test', test)
    assert len(periods) == len(par.planets['period']), "length of period arrays are not equal."
    np.testing.assert_array_almost_equal(test,np.zeros_like(test), decimal=1)
