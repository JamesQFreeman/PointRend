import numpy as np

def _if_near(point, mask, nearest_neighbor):
    nn = nearest_neighbor
    w,h = mask.shape[0],mask.shape[1]
    x,y = point
    mask = np.pad(mask,nn,'edge')
    x += nn
    y += nn
    if(w+nn>x and h+nn>y):
        x_i,y_i = int(x+0.5),int(y+0.5)
        #return True
        near = mask[x_i-nn:x_i+nn,y_i-nn:y_i+nn]
        if near.max()-near.min() != 0:
            if(x<w and y<h):
                return True
    return False


# ***
# *n* It's an example of 1-neighbor
# ***
#
# *****
# *****
# **n** It's an example of 2-neighbor
# *****
# *****
#
# Did you get any of that?
def _get_edge_k_neighbor(img,k):
    '''
    I will say the idea is identical to the
    the original _is_near, but this implement save the
    temporal result and thus speed up the whole
    process by a massive margin when a big amount of
    points requires calculation.

    This will return a array sized (w,h), 
    store the max-min value in its neighbor.
    '''
    w,h = img.shape
    padded = np.pad(img, k, 'edge')
    # this is the result image array
    res = np.zeros(img.shape)
    
    # This is the main process
    for i in range(w):
        for j in range(h):
            neighbor = padded[i:i+2*k,j:j+2*k]
            _max = neighbor.max()
            _min = neighbor.min()
            res[i-k,j-k] = (_max-_min)
    
    return res


def _new_if_near(point, edge_k_neighbor):
    x, y = point
    x, y = int(x), int(y)
    return edge_k_neighbor[x][y]>0


def getpoint(mask_img, k, beta, training = True, nearest_neighbor=3, new_if_near = True):
    w,h = mask_img.shape
    N = int(beta*k*w*h)
    xy_min = [0, 0]
    xy_max = [w-1, h-1]
    points = np.random.uniform(low=xy_min, high=xy_max, size=(N,2))
    #print(points)
    if(beta>1 or beta<0): 
        print("beta should be in range [0,1]")
        return NULL
    
    # for the training, the mask is a hard mask
    if training == True:
        if beta ==0: return points
        res = []
        if new_if_near:
            edge_k_neighbor = _get_edge_k_neighbor(mask_img,nearest_neighbor)
            for p in points:
                if _new_if_near(p,edge_k_neighbor):
                    res.append(p)
        else:
            for p in points:
                if _if_near(p,mask_img,nearest_neighbor):
                    res.append(p)

        others = int((1-beta)*k*w*h)
        not_edge_points = np.random.uniform(low=xy_min, high=xy_max, size=(others,2))
        for p in not_edge_points:
            res.append(p)
        return res
    
    # for the inference, the mask is a soft mask
    if training == False:
        res = []
        for i in range(w):
            for j in range(h):
                if mask_img[i,j] > 0:
                    res.append((i,j))
        return res
    