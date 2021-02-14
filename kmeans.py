from math import sqrt
from random import sample

import numpy as np
import cv2 as cv
from PIL import Image

def getimgdata(f, mode="L", dtype=np.float32):
    return np.asarray(Image.open(f).convert(mode), dtype=dtype)

def getnewimg(out, mode="L"):
    return Image.fromarray(out, mode)

def getcorners(img, n=100, q=0.001, md=25):
    return cv.goodFeaturesToTrack(img, n, q, md)

def kmeans(corners, k=3):
    clusters = initialize(corners, k)
    centroids = clusters.keys()
    clusters = assign(corners, clusters)
    clusters = update(clusters)
    while centroids != clusters.keys():
        centroids = clusters.keys()
        clusters = assign(corners, clusters)
        clusters = update(clusters)
    return assign(corners, clusters)

def initialize(corners, k=3):
    return {(c[0][0], c[0][1]): [] for c in sample(list(corners), k)}

def initialize_randompartition(corners, k=3):
    shuffle(corners)
    return {(0., 0.): corners[:34],
            (1., 1.): corners[34:67],
            (2., 2.): corners[67:]}

def assign(corners, clusters):
    for corner in corners:
        centroid = min([(distance(corner.flatten(), k), k) for k in clusters.keys()])[1]
        clusters[centroid].append(corner.flatten())
    return clusters

def update(clusters):
    return {tuple(sum(cluster) / len(cluster)): [] for cluster in clusters.values()}

def distance(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def imagecorners(img, clusters):
    out = img.copy()
    centroids = list(clusters.keys())
    red = np.array(clusters[centroids[0]], dtype=np.int32)
    green = np.array(clusters[centroids[1]], dtype=np.int32)
    blue = np.array(clusters[centroids[2]], dtype=np.int32)
    centroids = [list(map(round, c)) for c in centroids]
    for p in red:
        out[p[1]-1, p[0]] = np.array([255, 0, 0], dtype=np.uint8)
        out[p[1], p[0]-1] = np.array([255, 0, 0], dtype=np.uint8)
        out[p[1], p[0]] = np.array([255, 0, 0], dtype=np.uint8)
        out[p[1]+1, p[0]] = np.array([255, 0, 0], dtype=np.uint8)
        out[p[1], p[0]+1] = np.array([255, 0, 0], dtype=np.uint8)
    for p in green:
        out[p[1]-1, p[0]] = np.array([0, 255, 0], dtype=np.uint8)
        out[p[1], p[0]-1] = np.array([0, 255, 0], dtype=np.uint8)
        out[p[1], p[0]] = np.array([0, 255, 0], dtype=np.uint8)
        out[p[1]+1, p[0]] = np.array([0, 255, 0], dtype=np.uint8)
        out[p[1], p[0]+1] = np.array([0, 255, 0], dtype=np.uint8)
    for p in blue:
        out[p[1]-1, p[0]] = np.array([0, 0, 255], dtype=np.uint8)
        out[p[1], p[0]-1] = np.array([0, 0, 255], dtype=np.uint8)
        out[p[1], p[0]] = np.array([0, 0, 255], dtype=np.uint8)
        out[p[1]+1, p[0]] = np.array([0, 0, 255], dtype=np.uint8)
        out[p[1], p[0]+1] = np.array([0, 0, 255], dtype=np.uint8)
    for c in centroids:
        out[c[1]-1, c[0]-1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1]-1, c[0]] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1]-1, c[0]+1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1], c[0]] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1], c[0]-1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1], c[0]+1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1]+1, c[0]+1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1]+1, c[0]-1] = np.array([0, 0, 0], dtype=np.uint8)
        out[c[1]+1, c[0]] = np.array([0, 0, 0], dtype=np.uint8)
    return out

def boundingbox(img, clusters):
    out = img.copy()
    centroids = list(clusters.keys())
    red = np.array(clusters[centroids[0]], dtype=np.int32)
    red_x = [r[1] for r in red]    
    red_y = [r[0] for r in red]
    green = np.array(clusters[centroids[1]], dtype=np.int32)
    green_x = [g[1] for g in green]
    green_y = [g[0] for g in green]
    blue = np.array(clusters[centroids[2]], dtype=np.int32)
    blue_x = [b[1] for b in blue]
    blue_y = [b[0] for b in blue]
    for x in range(min(red_x), max(red_x)):
        out[x, min(red_y)] = [255, 0, 0]
        out[x, max(red_y)] = [255, 0, 0]
    for y in range(min(red_y), max(red_y)):
        out[min(red_x), y] = [255, 0, 0]
        out[max(red_x), y] = [255, 0, 0]
    for x in range(min(green_x), max(green_x)):
        out[x, min(green_y)] = [0, 255, 0]
        out[x, max(green_y)] = [0, 255, 0]
    for y in range(min(green_y), max(green_y)):
        out[min(green_x), y] = [0, 255, 0]
        out[max(green_x), y] = [0, 255, 0]
    for x in range(min(blue_x), max(blue_x)):
        out[x, min(blue_y)] = [0, 0, 255]
        out[x, max(blue_y)] = [0, 0, 255]
    for y in range(min(blue_y), max(blue_y)):
        out[min(blue_x), y] = [0, 0, 255]
        out[max(blue_x), y] = [0, 0, 255]
    return out

f = "image1.jpg"
color = getimgdata(f, mode="RGB", dtype=np.uint8)
gray = getimgdata(f)
corners = getcorners(gray)
clusters = kmeans(corners)
centroids = list(clusters.keys())
red = np.array(clusters[centroids[0]], dtype=np.int32)
green = np.array(clusters[centroids[1]], dtype=np.int32)
blue = np.array(clusters[centroids[2]], dtype=np.int32)

corners_out = imagecorners(color, clusters)
box_out = boundingbox(color, clusters)

getnewimg(corners_out, mode="RGB").show()
getnewimg(box_out, mode="RGB").show()

