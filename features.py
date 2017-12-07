import numpy as np
def prepare_features(item):
    img, mask, density, dims, rect = item
    #print(dims)
    sorted_dims = -np.sort(-np.array(dims))
    est_dims = [rect[2],rect[3],np.max(mask)*32]
    est_dims_volume = np.prod(est_dims)
    object_volume = np.sum(mask)
    dims_v = np.prod(dims)
    if est_dims_volume !=0:
        filled_proportion = object_volume/est_dims_volume
    else:
        filled_proportion = 0
    if (rect[2]*rect[3]) !=0:
        filled_proportion2d = np.sum(mask>(1.0/32.0))/(rect[2]*rect[3])
    else:
        filled_proportion2d=0

    aux = np.concatenate([
        [dims_v],#1
        [dims[0],dims[1],dims[2]],#3
        [est_dims[0],est_dims[1],est_dims[2]],#3 the length (1) and width (2) of the bounding box; the maximum entry in the thickness mask (3)
        # missing rect values are acutally the same as est_dims[0,1]
        [rect[0],rect[1],rect[4]], #3 the bounding rectangle’s center (4,5); and the bounding rectangle’s angle (6)
        [np.sum(mask)*32],#1 the sum of all values in the thickness mask (7)
        [np.sum(mask>(1.0/32.0))],#1 the number of non-zero entries in the thickness mask (8)
        [est_dims[0]*est_dims[1]],#2 the area of the bounding rectangle (9)
        [est_dims_volume],#1 the volume of the bounding box (10)
        [filled_proportion],#1  the proportion of voxels in the bounding box that are occupied by the object, o1 (11)
        [filled_proportion2d],#1  the proportion of pixels in the bounding rectangle that are occupied o2 (12)
        [dims_v*filled_proportion],#1 the 3D volume estimate (13) (o1 × L × W × H)
        [dims_v*filled_proportion2d],#1 and the 2D volume estimate (14) (o2 × L × W)
        [dims_v*filled_proportion**.5],#1
        [dims_v*filled_proportion2d**.5],#1
        [filled_proportion**2],#1
        [filled_proportion2d**2],#1
        sorted_dims,
        sorted_dims**2,
        [sorted_dims[0]*sorted_dims[1],sorted_dims[0]*sorted_dims[2],sorted_dims[1]*sorted_dims[2]],
        [sorted_dims[1]/sorted_dims[0],sorted_dims[2]/sorted_dims[0]],
        [est_dims[1]/est_dims[0],est_dims[2]/est_dims[0]],

        ])

    aux_input = []
    for a in aux:
        aux_input.append(a)
            
    x = (img, mask, np.array(aux_input))
    return x