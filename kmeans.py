#! /usr/bin/env python3

import sys
import os.path
import argparse
from math import sqrt
from random import sample, shuffle, randint

import numpy as np
import cv2 as cv
from PIL import Image


def getimgdata(f, mode="L", dtype=np.float32):
    return np.asarray(Image.open(f).convert(mode), dtype=dtype)


def getnewimg(out, mode="L"):
    return Image.fromarray(out, mode)


def getcorners(img, n, q, md):
    return cv.goodFeaturesToTrack(img, n, q, md)


def kmeans(corners, k, initializer):
    if initializer == "randompartition":
        clusters = randompartition(corners, k)
    else:
        clusters = forgy(corners, k)
    
    centroids = clusters.keys()
    clusters = assign(corners, clusters)
    clusters = update(clusters)
    while centroids != clusters.keys():
        centroids = clusters.keys()
        clusters = assign(corners, clusters)
        clusters = update(clusters)
    return assign(corners, clusters)


def forgy(corners, k):
    """Randomly select k points as initial centroids"""
    return {(c[0][0], c[0][1]): [] for c in sample(list(corners), k)}


def randompartition(corners, k):
    """Randomly assign all points to k clusters and update"""
    shuffle(corners)
    return {tuple(sum(corners[:34])/len(corners[:34])): corners[:34],
            tuple(sum(corners[34:67])/len(corners[34:67])): corners[34:67],
            tuple(sum(corners[67:])/len(corners[67:])): corners[67:]}


def assign(corners, clusters):
    """Assign found corners to nearest centroid"""
    for corner in corners:
        centroid = min([(distance(corner.flatten(), k), k)
                        for k in clusters.keys()])[1]
        clusters[centroid].append(corner.flatten())
    return clusters


def update(clusters):
    """Update centroid position"""
    return {tuple(sum(cluster) / len(cluster)): []
            for cluster in clusters.values()}


def distance(x, y):
    """Euclidean Distance"""
    return sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


def imagecorners(img, clusters):
    """"Draws the clustered corners in different colors onto original image"""
    out = img.copy()
    centroids = list(clusters.keys())
    for centroid in centroids:
        c = list((map(round, centroid)))
        out[c[1]-1, c[0]-1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1]-1, c[0]] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1]-1, c[0]+1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1], c[0]] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1], c[0]-1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1], c[0]+1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1]+1, c[0]+1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1]+1, c[0]-1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1]+1, c[0]] = np.array([0, 0, 0], dtype=np.uint8)
        cluster = np.array(clusters[centroid], dtype=np.int32)
        color = [randint(0, 255), randint(0, 255), randint(0, 255)]
        for point in cluster:
            out[point[1]-1, point[0]] = np.array(color, dtype=np.uint8)
            out[point[1], point[0]-1] = np.array(color, dtype=np.uint8)
            out[point[1], point[0]] = np.array(color, dtype=np.uint8)
            out[point[1]+1, point[0]] = np.array(color, dtype=np.uint8)
            out[point[1], point[0]+1] = np.array(color, dtype=np.uint8)
    return out


def boundingbox(img, clusters):
    """Draws k bounding boxes around corner clusters over original image"""
    out = img.copy()
    centroids = list(clusters.keys())
    for c in centroids:
        color = [randint(0, 255), randint(0, 255), randint(0, 255)]
        points = np.array(clusters[c], dtype=np.int32)
        px = [p[1] for p in points]
        py = [p[0] for p in points]
        for x in range(min(px), max(px)):
            out[x, min(py)] = color
            out[x, max(py)] = color
        for y in range(min(py), max(py)):
            out[min(px), y] = color
            out[max(px), y] = color
    return out


def setup_argparser():
    parser = argparse.ArgumentParser(
        description="Detect strongest edges and draw bounding box"
    )
    parser.add_argument(
        "-f", "--file",
        help="path to image file",
        dest="file",
        required=False
    )
    parser.add_argument(
        "-b", "--bounding-box",
        help="Draw bounding box around clusters",
        dest="box",
        action="store_true",
        required=False
    )
    parser.add_argument(
        "-p", "--points",
        help="Draw corner clusters",
        dest="points",
        action="store_true",
        required=False
    )
    parser.add_argument(
        "-k", "--clusters",
        help="number of clusters",
        metavar="N",
        dest="k",
        type=int,
        default=3,
        required=False
    )
    parser.add_argument(
        "-c", "--corners",
        help="number of corners",
        metavar="N",
        dest="corners",
        type=int,
        default=100,
        required=False
    )
    parser.add_argument(
        "-q", "--quality",
        help="corner quality threshold",
        metavar="F",
        dest="quality",
        type=float,
        default=0.001,
        required=False
    )
    parser.add_argument(
        "-m", "--minimum-distance",
        help="Minimum distance apart from other corners",
        dest="md",
        type=int,
        default=25,
        required=False
    )
    parser.add_argument(
        "-r", "--random-partition",
        help="Random Partition initializer: Assign each observation to a group randomly",
        dest="rp",
        action="store_true",
        required=False
    )
    return parser




if __name__ == '__main__':
    args = setup_argparser().parse_args()

    f = "image1.jpg"
    if args.file:
        f = args.file
    
    init = "forgy"
    if args.rp:
        init = "randompartition"

    both = True
    if args.box or args.points:
        both = False

    color = getimgdata(f, mode="RGB", dtype=np.uint8)
    gray = getimgdata(f)
    corners = getcorners(gray, args.corners, args.quality, args.md)
    clusters = kmeans(corners, args.k, init)

    if both or args.points:
        getnewimg(imagecorners(color, clusters), mode="RGB").show()
    
    if both or args.box:
        getnewimg(boundingbox(color, clusters), mode="RGB").show()

    sys.exit(0)