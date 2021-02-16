CIS 666, Spring 2021
Bradley Dowling 
CSU ID: 2657649

Assignment 2: Feature Detection
---


### Setup

* Run the `setup` script in the root of the project directory: makes a virtual environment, updates pip, and installs dependencies

* Tested with:
    * Ubuntu 20.04 x86_64
    * Python 3.8.5
    * pip 21.0.1
    * numpy 1.20.1
    * opencv-python 4.5.1
    * Pillow 8.1.0

---

### Usage

* Naive K-means algorithm (Lloyd's Algorithm). 
* Default initialization method is Forgy (randomly select k observations as initial centroids).

* Switch to virtual environment with the shell command `source bin/activate`.
* `python kmeans.py` with no arguments will draw the 100 strongest corners in k=3 clusters and bounding boxes on `image1.jpg`.
* `./kmeans.py` should also work.
* `./kmeans.py -h` will show options and switches.
    * -h, --help:                   show help
    * -f, --file:                   select a different file
    * -k, --clusters [k]:           integer, number of clusters
    * -c, --corners [n]:            integer, n strongest corners
    * -q, --quality [f]:            float, corner quality threshold
    * -m, --minimum-distance [n]:   integer, minimum distance from other corners
    * -B, --blocksize [n]:          integer, blocksize for Harris corner detector
    * -K, --free-parameter [f]:     float, free parameter k for Harris corner detector
    * -r, --random-partition:       initialize by calculating initial centroids by randomly assigning each point to k partitions
    * -p, --points:                 draw corner points onto original image, randomly colored by cluster
    * -b, --bounding-box:           draw bounding boxes onto original image
    * -s, --shi-tomasi:             Use Shi-Tomasi corner detection instead of Harris

---

### Part 1: Find Corners

* The Pillow class `Image`, and numpy functions are used to convert a color image to a grayscale `ndarray` object.
* The OpenCV function goodFeaturesToTrack() is used to collect the corner observations.
* The parameters for number of corners, quality, and minimum distance, blocksize, and 
free parameter K can be selected at the command line.
* Harris corner detection is used by default. The `-s` switch will change to Shi-Tomasi corner detection.

```python
corners = goodFeaturesToTrack(gray, args.corners, args.quality, 
                                  args.md, blockSize=args.blocksize, 
                                  useHarrisDetector=args.st, k=args.K)
```

---

### Part 2: K-means

* Here, I implement naive K-means using Lloyd's Algorithm. 
* Numpy ndarrays are used for performance, but no other external library code is used.

* The K-means algorithm must have an initialization step. I implemented two popular initialization methods:

    * **Forgy method**: Select k random observations and select them as the initial centroids.
    
```python
def forgy(corners, k):
    """Randomly select k points as initial centroids"""
    return {(c[0, 0], c[0, 1]): [] for c in sample(list(corners), k)}
```

   * **Random Partitioning**: Randomly assign all observations to k clusters, then calculate the intial centroids.

```python
def randompartition(corners, k):
    """Randomly assign all points to k clusters and calculate centroids"""
    clusters = {}
    shuffle(corners)
    indices = list(linspace(0, len(corners), num=k+1, dtype=int))
    for start, end in zip(indices[:len(indices)-1], indices[1:]):
        centroid = (sum(corners[start:end]) / len(corners[start:end]))[0]
        clusters[(centroid[0], centroid[1])] = []
    return clusters
```

* The K-means algorithm switches between two operations:

    * **assignment**: assign observations to *k* clusters based on the minimal distance from that cluster's centroid. 

```python
def assign(corners, clusters):
    """Assign found corners to nearest centroid"""
    for corner in corners:
        centroid = min([(distance(corner.flatten(), k), k)
                        for k in clusters.keys()])[1]
        clusters[centroid].append(corner.flatten())
    return clusters

```

   * **update**: recalculate the centroids based on the mean of the points assigned to the cluster.

```python
def update(clusters):
    """Update centroid position"""
    return {tuple(sum(cluster) / len(cluster)): []
            for cluster in clusters.values()}

```

* K-means terminates when the centroids no longer change position. It is not guaranteed to be globally optimal. In some cases, the clusters are a very poor reflection of the test image, but most of the time it performs well. The main line of the algorithm follows:

```python
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
```

---

### Part 3: Bounding Box

* Bounding boxes are drawn around the clusters based on the points with the maximum vertical and horizontal distances from their centroid.
* The Shi-Tomasi corner detector generally outperformed the Harris-Stephens corner detector.

```python
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
```
---
