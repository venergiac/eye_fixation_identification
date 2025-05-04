import numpy as np
from scipy.signal import savgol_filter

def extract_features(x,y,t, fixations, min_duration=5):

    ret = []
    _prev_fix= 0
    _x,_y,_t,_i = [],[],[],[]
    for _idx , _fix in enumerate(fixations):
        if _fix == 1:
            _x.append(x[_idx])
            _y.append(y[_idx])
            _t.append(t[_idx])
            _i.append(_idx)
            
        if _prev_fix != _fix:
            if len(_i) > min_duration:
                ret.append([np.mean(_x), np.mean(_y), np.min(_t), np.max(_t), np.min(_i), np.max(_i)] )
            _x,_y,_t,_i = [],[],[],[]
        
        _prev_fix = _fix

    return np.array( ret )

def extract_fixations(x,y,t, th=0.01, window=10, method="cov"):

    if len(x) != len(y):
        raise Exception("X and Y musta have the same dimension")

    x_ = savgol_filter(x, window*2, 2)
    y_ = savgol_filter(y, window*2, 2)
    xy =  np.column_stack((x_, y_))

    th_ = th * max( np.max(x), np.max(y) ) 

    fixations = np.zeros((xy.shape[0]))
    for i in range(window, xy.shape[0] - window):

        decision = False
        if method=="cov":
            cov = np.cov(xy[i:i+window, 0], xy[i:i+window, 1])
            decision = np.abs( cov[0,1] ) < th_
        else:
            var_x = xy[i:i+window, 0].var()
            var_y = xy[i:i+window, 1].var()
            decision = np.abs(var_x - var_y) < th_

        fixations[i] = 1 if decision else 0
        
    return fixations