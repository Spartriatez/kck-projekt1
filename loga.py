#!/usr/bin/python3
import matplotlib.pyplot as plt
import os
import logging
import numpy as np
from multiprocessing import Pool
from skimage import data, morphology
from skimage import transform as tf
from scipy import ndimage
from skimage import measure
from skimage.feature import (canny, match_descriptors, ORB, plot_matches)

descriptor_extractor = ORB()

nazwy = ['Algida','Allegro','Apple','Audi','Baidu','Bentley','Biedronka','Carrefour','Dell','Facebook','Fiat','Google','Heyah','Hyundai','Kia','Lidl','Mazda','Nasa','NFZ','Opel','Orlen','PKO BP','Poznan','PP','Roshen','Skoda','Sony','UAM','Ubuntu','Vizir','Vizir','Wedel','MS Windows','Winiary']

cards=[]

def load_pattenrs(filename):
    z_patterns = []
    print ('Load pattern: ' + filename)
    img = data.imread('logo/' + filename, as_grey=True)
    tmp = img
    tmp = canny(tmp, sigma=2.0)
    tmp = morphology.dilation(tmp, morphology.disk(2))
    descriptor_extractor.detect_and_extract(tmp)
    obj_key = descriptor_extractor.keypoints
    obj_desc = descriptor_extractor.descriptors
    z_patterns.append([img, obj_desc, obj_key, filename])
    return z_patterns


def load_scenes(filename):
    zipped_scenes = []
    """ print ('Picture to compare: ' + filename)
    img = data.imread('foto/' + filename, as_grey=True)
    tmp = img
    tmp = canny(tmp, sigma=2.0)
    tmp = ndimage.binary_fill_holes(tmp)
    tmp = morphology.remove_small_objects(tmp, 2000)
    contours = measure.find_contours(tmp, 0.8)
    ymin, xmin = contours[0].min(axis=0)
    ymax, xmax = contours[0].max(axis=0)
    if xmax - xmin > ymax - ymin:
        xdest = 1000
        ydest = 670
    else:
        xdest = 670
        ydest = 1000
    src = np.array(((0, 0), (0, ydest), (xdest, ydest), (xdest, 0)))
    dst = np.array(((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)))
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, dst)
    warped = tf.warp(img, tform3, output_shape=(ydest, xdest))
    tmp = canny(tmp, sigma=2.0)
    tmp = morphology.dilation(tmp, morphology.disk(2))
    descriptor_extractor.detect_and_extract(tmp)
    obj_key = descriptor_extractor.keypoints
    scen_desc = descriptor_extractor.descriptors
"""
    print ('Load pattern: ' + filename)
    img = data.imread('foto/' + filename, as_grey=True)
    tmp = img
    tmp = canny(tmp, sigma=2.0)
    tmp = morphology.dilation(tmp, morphology.disk(2))
    descriptor_extractor.detect_and_extract(tmp)
    obj_key = descriptor_extractor.keypoints
    obj_desc = descriptor_extractor.descriptors
    zipped_scenes.append([tmp,obj_desc, obj_key, filename])
    return zipped_scenes


def set_names(size):
    i = -1
    while i < size:
        if i > -1 and i < len(nazwy):
            cards.append(nazwy[i] + ' ')
        i += 1


def recognize(pattern, name, scene):
    zipped_matches = []
    match = match_descriptors(scene, pattern, cross_check=True, max_distance=0.5)
    zipped_matches.append([match.size, name, match])
    return zipped_matches

def f_wrap(arg_list):
    try:
        return recognize(*arg_list)
    except Exception:
        logging.exception("f(%r) failed" % (arg_list,))

def main():
    p = Pool(5)
    listing = os.listdir('logo')
    zipped_patterns = p.map(load_pattenrs, listing)
    zipped_patterns = [ent for sublist in zipped_patterns for ent in sublist]
    listing = os.listdir('foto')
    zipped_scenes = p.map(load_scenes, listing)
    zipped_scenes = [ent for sublist in zipped_scenes for ent in sublist]
    zipped_patterns.sort(key=lambda x: x[3])
    p_img, patterns, p_key, tmp = zip(*zipped_patterns)
    zipped_scenes.sort(key=lambda x: x[3])
    s_img, scenes, s_key, tmp = zip(*zipped_scenes)
    set_names(len(zipped_patterns) - 1)
    k = 0
    for j in scenes:
        arg_list = []
        for a, b, c, d in zipped_patterns:
            arg_list.append([b, d, j])
        zipped_matches = p.map(f_wrap, arg_list)
        zipped_matches = [ent for sublist in zipped_matches for ent in sublist]
        zipped_matches.sort(key=lambda x: x[1])
        matches, tmp, m_array = zip(*zipped_matches)
        best_match = max(matches)
        proc = 1.0
        id = 0
        result = 'Logo to: '
        for i in range(len(patterns)):
            el = abs((patterns[i].size/j.size) - 1)
            if matches[i] == best_match:
                result += cards[i] + " "
            if matches[i] == best_match and el < proc:
                proc = el
                id = i
        fig, ax = plt.subplots()
        plt.gray()
        plot_matches(ax, p_img[id], s_img[k], p_key[id], s_key[k], m_array[id])
        ax.axis('off')
        plt.show()
        print (result)
        k += 1

if __name__ == '__main__':
    main()
