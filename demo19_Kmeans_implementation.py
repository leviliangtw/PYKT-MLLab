from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

# np.random.randn: “normal” (Gaussian) distribution of mean 0 and variance 1
part1 = np.random.randn(50, 2) + [2, 2]
part2 = np.random.randn(50, 2) + [0, -2]
part3 = np.random.randn(50, 2) + [-2, 2]

X = np.r_[part1, part2, part3]
# print(type(X), X.shape)
# [plt.scatter(element[0], element[1], c='black', s=7) for element in X]
# for e in X:
#     plt.scatter(e[0], e[1], c='black', s=7)
# plt.show()

k = 3
C_X = np.random.randint(np.min(X), np.max(X), size=k)
C_y = np.random.randint(np.min(X), np.max(X), size=k)
C = np.array(list(zip(C_X, C_y)), dtype=np.float32)
# print(C)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

p = np.array([[0, 0]])
q = np.array([[3, 4]])
print(dist(p, q))

C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
delta = dist(C, C_old, None)
print(delta)


def plot_kmean(current_cluster, delta):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots()
    # for each color(cluster)
    for i in range(k):
        # get current cluster
        points = np.array([X[j] for j in range(len(X))
                           if current_cluster[j] == i])
        # scater by x, y coordinate
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    # plot 3 center
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#C0FFEE')
    plt.title('delta will be: %4f' % delta)
    # plt.plot()
    plt.show()


while delta != 0:
    for i in range(len(X)):
        # calculate distance from current point to 3 k-mean center
        distance = dist(X[i], C)
        cluster = np.argmin(distance)
        clusters[i] = cluster
    C_old = deepcopy(C)
    # move k-mean center to current point
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        # calculate new center
        C[i] = np.mean(points, axis=0)
    # calculate distance between old <----> new center
    delta = dist(C, C_old, None)
    plot_kmean(clusters, delta)
