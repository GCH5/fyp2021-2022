import numpy as np
# import scipy.spatial as scipy_spatial
import scipy.ndimage as scipy_ndimage
import os
import glob
from matplotlib import pyplot as plt
from tqdm import tqdm

import hdf5storage


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


def run():
    # Generate density map with fixed kernel.
    root = './UCF-QNRF_ECCV18/'
    train = os.path.join(root, 'Train')
    test = os.path.join(root, 'Test')
    path_sets = [train, test]

    img_paths = []
    for path in path_sets:
        img_paths.extend(glob.glob(os.path.join(path, '*.jpg')))
    index = 0

    for img_path in tqdm(img_paths):
        print(img_path)
        img_shape = plt.imread(img_path).shape
        mat = hdf5storage.loadmat(img_path.replace('.jpg', '_ann.mat'))
        points = mat['annPoints']
        index = index + 1
        dmap = gaussian_filter_density(img_shape, points)

        np.save(img_path.replace('.jpg', '.npy'), dmap)
        if(index > 2):
            break
        # misc.imsave(img_path.replace('images','GT'), k)


if __name__ == "__main__":
    run()
