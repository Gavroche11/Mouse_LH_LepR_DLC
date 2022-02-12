import numpy as np
import pandas as pd


# This function receives a list of distinct integers in the increasing order,
# and returns a list of lists each of which consist of successive integers.
# successive([1, 2, 3, 8, 9, 11, 13])
# >>> [[1, 2, 3], [8, 9], [11], [13]]

def successive(frames):
    result = list()
    i = 0
    if len(frames) == 0:
        return list()
    else:
        while i in range(len(frames)):
            if i == 0:
                newlist = [frames[i]]
            elif frames[i] - frames[i - 1] == 1:
                newlist.append(frames[i])
            else:
                result.append(newlist)
                newlist = [frames[i]]
            i += 1
        result.append(newlist)
        return result

# This function receives a list of distinct integers in the increasing order,
# and returns a list of the first / middle / last element of each partition of successive integers.
# representation([1, 2, 3, 8, 9, 11, 13], x=0)
# >>> [1, 8, 11, 13]
# representation([1, 2, 3, 8, 9, 11, 13], x=1)
# >>> [2, 8, 11, 13]
# representation([1, 2, 3, 8, 9, 11, 13], x=2)
# >>> [3, 9, 11, 13]

def representation(frames, x=1):
    assert x == 0 or x == 1 or x == 2, 'Input 0, 1, or 2 for first, middle, or last frame respectively'
    sf = successive(frames)
    result = list()
    for i, j in enumerate(sf):
        if x == 0:
            result.append(j[0])
        elif x == 1:
            result.append(round((j[0] + j[-1]) / 2))
        else:
            result.append(j[-1])
    return result

# This function processes the raw output from DLC based on the likelihood.
# We remove the values with likelihood lower than our criterion, and interpolate those empty values linearly.

def lininterpol(df, bodyparts, ll_crit, absolute=True):
    '''
    df: DataFrame from pd.read_csv(filepath, header=[1, 2], index_col=0, skiprows=0)
    bodyparts: list of bodyparts of interest
    ll_crit: real number in [0, 1) or list of real numbers
    absolute: bool or list of bools
    '''
    numbodyparts = len(bodyparts)
    numframes = len(df)

    values = df[bodyparts].values.reshape(-1, numbodyparts, 3).transpose([1, 0, 2]) # np array of shape (numbodyparts, numframes, 3)
    
    if type(absolute) == bool:
        absolute = [absolute] * numbodyparts
    
    if type(ll_crit) == float:
        ll_crit = [ll_crit] * numbodyparts

    mins = []
    for i in range(numbodyparts):
        if absolute[i]:
            mins.append(ll_crit[i])
        else:
            cutoff_index = values[i, :, -1].argsort()[int(numframes * ll_crit[i])]
            mins.append(values[i, cutoff_index, -1])

    mins = np.array(mins).reshape(-1, 1) # np array of shape (numbodyparts, 1)
    
    good = values[:, :, -1] >= mins # np array of shape (numbodyparts, numframes), T or F

    assert good.all(axis=0).sum() >= 2, 'Likelihood too high'

    start, end = np.where(good.all(axis=0))[0][0], np.where(good.all(axis=0))[0][-1]

    values = values[:, start:(end + 1), :] # np array of shape (numbodyparts, # frames for use, 3)
    good = good[:, start:(end + 1)] # np array of shape (numbodyparts, # frames for use)

    for i in range(numbodyparts):
        bad0 = np.array(representation(np.where(~good[i])[0], x=0)).reshape(-1, 1)
        bad1 = np.array(representation(np.where(~good[i])[0], x=2)).reshape(-1, 1)
        bads = np.concatenate((bad0, bad1), axis=1)

        for j in range(bads.shape[0]):
            prev_frame = int(bads[j, 0] - 1)
            next_frame = int(bads[j, 1] + 1)
            values[i, prev_frame:next_frame, :-1] = np.linspace(values[i, prev_frame, :-1], values[i, next_frame, :-1], num=(next_frame - prev_frame), endpoint=False)

    tuples = []
    for bp in bodyparts:
        tuples.append((bp, 'x'))
        tuples.append((bp, 'y'))


    new_df = pd.DataFrame(values[:, :, :-1].transpose([1, 0, 2]).reshape(-1, 2 * numbodyparts), columns=pd.MultiIndex.from_tuples(tuples))

    '''
    returns:
        (new_df of index=df.index, columns consisting of (bodypart, x) and (bodypart, y))
        start,
        end
    '''
    return new_df, start, end

# This function merges two DLC raw outputs (e.g., videos from side and bottom) so that they share the same index.
# We need to give in the timestamps (time0, time1) from two videos which represent the same moment in the real world.

def merge(coords0, coords1, start0, start1, time0, time1, FPS0, FPS1, name='side'):
    crit_index0 = int(FPS0*time0) - start0
    crit_index1 = int(FPS1*time1) - start1
    
    newcols = []
    for col in list(coords1.columns):
        newcols.append((col[0]+f'_{name}', col[1]))

    transform = (np.array(coords0.index) - crit_index0)*(FPS1/FPS0) + crit_index1
    q = np.floor(transform).astype(int)
    r = transform - q
    
    def transform_coords(arr):
        i = arr[0]
        if 0 <= q[i] < coords1.index[-1]:
            return coords1.values[q[i]] * (1 - r[i]) + coords1.values[q[i] + 1] * r[i]
        elif q[i] == coords1.index[-1] and r[i] == 0.0:
            return coords1.values[q[i]]
        else:
            return np.full(shape=(len(newcols),), fill_value=np.nan)
    
    new = pd.DataFrame(data=np.apply_along_axis(transform_coords, 1, np.array(coords0.index).reshape(-1, 1)),
                       index=coords0.index, columns=pd.MultiIndex.from_tuples(newcols))
    
    coords = pd.concat([coords0, new], axis=1).dropna()
    start = start0 + coords.index[0]
    end = start + len(coords.index) - 1

    coords = coords.reset_index(drop=True)

    return coords, start, end

# The following four functions return whether each frame satisfies our conditions.
# The frames satisfying all of our four conditions are defined as 'Evaluation or Swallowing.'

# This function determines whether [the distance between hands < the distance between paws].

def compare_hand_paw(new_df):
    '''
    lininterpoled df
    columns must contain ['hand_L', 'hand_R', 'paw_L', 'paw_R']
    '''
    hand_dist = np.linalg.norm(new_df['hand_L'].values - new_df['hand_R'].values, axis=1)
    paw_dist = np.linalg.norm(new_df['paw_L'].values - new_df['paw_R'].values, axis=1)

    result = hand_dist < paw_dist
    '''
    returns 1D np array consisting of bool
    '''
    return result

# This function determines whether [y-coordinate of the tail base < y-coordinate of the middlepoint of two paws < y-coordinate of the middlepoint of two hands].

def compare_ys(new_df):
    '''
    lininterpoled df
    columns must contain ['tail_base', 'hand_L', 'hand_R', 'paw_L', 'paw_R']
    '''
    tb_y = new_df[('tail_base', 'y')].values
    paws_y = new_df[[('paw_L', 'y'), ('paw_R', 'y')]].values.mean(axis=1)
    hands_y = new_df[[('hand_L', 'y'), ('hand_R', 'y')]].values.mean(axis=1)

    result = (tb_y < paws_y) & (paws_y < hands_y)
    '''
    returns 1D np array consisting of bool
    '''
    return result

# This function determines whether the snout is inside the food zone (trapezoid) when viewed from bottom.

def foodzone_bottom(new_df, vertices):
    '''
    lininterpoled df
    columns must contain ['snout']  
    '''
    '''
    0    1
    2    3 
    '''
    snout = new_df['snout'].values
    
    def determine_fz(point):
        x, y = point
        if y < vertices[0, 1] or y > vertices[2, 1]:
            return False
        else:
            ratio = (y - vertices[0, 1]) / (vertices[2, 1] - vertices[0, 1])
            left_x = vertices[0, 0] + ratio * (vertices[2, 0] - vertices[0, 0])
            right_x = vertices[1, 0] + ratio * (vertices[3, 0] - vertices[1, 0])
            return x >= left_x and x <= right_x
    
    return np.apply_along_axis(determine_fz, 1, snout)

# This function determines whether the snout is inside the food zone (trapezoid) when viewed from side.

def foodzone_side(new_df, y0, y1):
    snout_height = new_df[('snout_side', 'y')].values
    return (snout_height >= y0) & (snout_height <= y1)

conditions = [compare_hand_paw, compare_ys, foodzone_bottom, foodzone_side]

# Following lists are the bodyparts which must be contained in the DLC output of bottom-view and side-view, respectively.

required_bps_bottom = ['tail_base', 'hand_L', 'hand_R', 'paw_L', 'paw_R', 'snout']
required_bps_side = ['snout']
