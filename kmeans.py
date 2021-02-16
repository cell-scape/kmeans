#! /usr/bin/env python3

from sys import exit
from os.path import isfile, abspath
from argparse import ArgumentParser
from math import sqrt, dist
from random import sample, shuffle, randint
from numpy import linspace, ndarray, array, asarray, uint8, int32, float32
from cv2 import goodFeaturesToTrack
from PIL import Image


def getimgdata(f, mode="L", dtype=float32):
    return asarray(Image.open(f).convert(mode), dtype=dtype)


def getnewimg(out, mode="RGB"):
    return Image.fromarray(out, mode)


def kmeans(corners, k, rp=False):
    if rp:
        clusters = randompartition(corners, k)
    else:
        clusters = forgy(corners, k)
    
    prev_centroids = clusters.keys()
    clusters = assign(corners, clusters)
    clusters = update(clusters)
    while prev_centroids != clusters.keys():
        prev_centroids = clusters.keys()
        clusters = assign(corners, clusters)
        clusters = update(clusters)
    return assign(corners, clusters)


def forgy(corners, k):
    """Randomly select k points as initial centroids"""
    return {(c[0, 0], c[0, 1]): [] for c in sample(list(corners), k)}


def randompartition(corners, k):
    """Randomly assign all points to k clusters and update centroids"""
    clusters = {}
    shuffle(corners)
    indices = list(linspace(0, len(corners), num=k+1, dtype=int))
    for start, end in zip(indices[:len(indices)-1], indices[1:]):
        centroid = (sum(corners[start:end]) / len(corners[start:end]))[0]
        clusters[(centroid[0], centroid[1])] = []
    return clusters


def assign(corners, clusters):
    """Assign found corners to nearest centroid"""
    for corner in corners:
        centroid = min([(dist(corner.flatten(), k), k)
                        for k in clusters.keys()])[1]
        clusters[centroid].append(corner.flatten())
    return clusters


def update(clusters):
    """Update centroid position"""
    return {tuple(sum(cluster) / len(cluster)): []
            for cluster in clusters.values()}


def imagecorners(img, clusters):
    """"Draws the clustered corners in different colors onto original image"""
    out = img.copy()
    for centroid in clusters.keys():
        c = tuple(map(round, centroid))
        out[c[1]-1, c[0]-1] = array([0, 0, 0], dtype=uint8)
        out[c[1]-1, c[0]] = array([0, 0, 0], dtype=uint8)
        out[c[1]-1, c[0]+1] = array([0, 0, 0], dtype=uint8)
        out[c[1], c[0]] = array([0, 0, 0], dtype=uint8)
        out[c[1], c[0]-1] = array([0, 0, 0], dtype=uint8)
        out[c[1], c[0]+1] = array([0, 0, 0], dtype=uint8)
        out[c[1]+1, c[0]+1] = array([0, 0, 0], dtype=uint8)
        out[c[1]+1, c[0]-1] = array([0, 0, 0], dtype=uint8)
        out[c[1]+1, c[0]] = array([0, 0, 0], dtype=uint8)
        cluster = array(clusters[centroid], dtype=int32)
        color = [randint(0, 255), randint(0, 255), randint(0, 255)]
        for point in cluster:
            out[point[1]-1, point[0]] = array(color, dtype=uint8)
            out[point[1], point[0]-1] = array(color, dtype=uint8)
            out[point[1], point[0]] = array(color, dtype=uint8)
            out[point[1]+1, point[0]] = array(color, dtype=uint8)
            out[point[1], point[0]+1] = array(color, dtype=uint8)
    return out


def boundingbox(img, clusters):
    """Draws k bounding boxes around corner clusters over original image"""
    out = img.copy()
    centroids = list(clusters.keys())
    for c in centroids:
        color = [randint(0, 240), randint(0, 240), randint(0, 240)]
        points = array(clusters[c], dtype=int32)
        px = [p[1] for p in points]
        py = [p[0] for p in points]
        for x in range(min(px), max(px)):
            out[x, min(py)] = color
            out[x, max(py)] = color
        for y in range(min(py), max(py)):
            out[min(px), y] = color
            out[max(px), y] = color
    return out


def parser():
    parser = ArgumentParser(
        description="Detect n strongest corners and draw k bounding boxes"
    )
    parser.add_argument(
        "-f", "--file",
        help="path to image file",
        dest="file",
        required=False
    )
    parser.add_argument(
        "-b", "--bounding-box",
        help="Draw bounding box",
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
        metavar="int",
        dest="k",
        type=int,
        default=3,
        required=False
    )
    parser.add_argument(
        "-c", "--corners",
        help="number of corners",
        metavar="int",
        dest="corners",
        type=int,
        default=100,
        required=False
    )
    parser.add_argument(
        "-q", "--quality",
        help="corner quality threshold",
        metavar="float",
        dest="quality",
        type=float,
        default=0.001,
        required=False
    )
    parser.add_argument(
        "-m", "--minimum-distance",
        help="Minimum distance apart from other corners",
        metavar="int",
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
    parser.add_argument(
        "-s", "--shi-tomasi",
        help="Use Shi-Tomasi Corner Detector instead of Harris-Stephens",
        dest="st",
        action="store_false",
        required=False
    )
    parser.add_argument(
        "-B", "--block-size",
        help="Block size for corner detector",
        metavar="int",
        dest="blocksize",
        type=int,
        default=4,
        required=False
    )
    parser.add_argument(
        "-K", "--free-parameter",
        help="free parameter k for corner detector",
        metavar="float",
        dest="K",
        type=float,
        default=0.04,
        required=False
    )
    return parser


if __name__ == '__main__':
    args = parser().parse_args()

    f = "image1.jpg"
    if args.file and isfile(abspath(args.file)):
        f = abspath(args.file)

    color = getimgdata(f, mode="RGB", dtype=uint8)
    gray = getimgdata(f)
    corners = goodFeaturesToTrack(gray, args.corners, args.quality, 
                                  args.md, blockSize=args.blocksize, 
                                  useHarrisDetector=args.st, k=args.K)
    clusters = kmeans(corners, args.k, args.rp)

    if (args.points and args.box) or (not args.points and not args.box):
        getnewimg(boundingbox(imagecorners(color, clusters), clusters)).show()
    elif args.points:
        getnewimg(imagecorners(color, clusters)).show()
    elif args.box:
        getnewimg(boundingbox(color, clusters)).show()
    exit(0)