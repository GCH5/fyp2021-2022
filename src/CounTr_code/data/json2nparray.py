
from pathlib import Path
import json
import numpy as np
import numpy as np
import scipy.ndimage as scipy_ndimage
from matplotlib import pyplot as plt
from tqdm import tqdm




def gaussian_filter_density(img_shape, points):
    '''
    This code generate GT for Counting-Bench use k-nearst, will take one minute or more to generate a density-map with one thousand people.

    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col).

    return:
    density: the GT density-map we want. Same shape as input image but only has one channel.
    '''
    img_shape = [img_shape[0], img_shape[1]]
    print("Shape of current image: ", img_shape,
          ". Totally need generate ", len(points), "gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    # leafsize = 2048
    # build kdtree
    # tree = scipy_spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    # distances, locations = tree.query(points, k=4)

    print('generate density...')
    for pt in tqdm(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.  # 1.
        else:
            continue
        # if gt_count > 1:
        #     sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        # else:
        #     sigma = np.average(np.array(img_shape))/2./2. #case: 1 point
        sigma = 5  # 3 for worldexpo & Mall dataset
        density += scipy_ndimage.filters.gaussian_filter(
            pt2d, sigma, mode='constant')

    print('done.')

    return density






for path in Path('sequence01_03').rglob('*.json'):
    with open(path, 'r') as f:
        data = json.load(f)
        points = np.array(list(map(lambda shape: shape['points'][0], data['shapes'])))
        img_shape = plt.imread(str(path).replace('.json','.jpg')).shape
        dmap = gaussian_filter_density(img_shape, points)

        np.save(str(path).replace('.json', '.npy'), dmap)
        # misc.imsave(img_path.replace('images','GT'), k)