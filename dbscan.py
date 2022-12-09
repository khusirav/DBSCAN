import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from sklearn import datasets
from sklearn.metrics.cluster import v_measure_score
from matplotlib.ticker import FuncFormatter




def euclid(dot1, dot2):
    return np.linalg.norm(dot1 - dot2)

#создание карты расстояний для ускорения работы алгоритма
def simple_distmap(dots, distance_func):
    distmap = np.zeros((dots.shape[0], dots.shape[0]))
    for i in range(dots.shape[0]):
        for j in range(dots.shape[0]):
            distmap[i, j] = distance_func(dots[i], dots[j])
    return distmap

#метод транссформации разметки получившихся кластеров для применения методов оценки точности кластеризации из sklearn
def label_clusters(dataset, clusters_dict):
    cluster_labels = []
    for i in range(dataset[1].shape[0]):
        cluster_labels.append(0)
        
    for cluster_num in range(len(clusters_dict)):
        for point in clusters_dict.get(cluster_num):
            point_index = np.where(dataset[0] == np.array(point))[0][0]
            cluster_labels[point_index] = int(cluster_num)

    return cluster_labels


#метод , реализующий алгоритм DBSCAN
def DBSCAN(eps, min_neighbours, distance_func, dataset, visualize):

    NOISE = 0
    C = 0

    if visualize:
        plt.ion()
        plt.scatter(dataset[:, 0], dataset[:, 1], color = 'b')

    visited_points = set()
    clustered_points = set()
    clusters = {NOISE: []}
    colors = ("k" ,"r", "g", "b", "c", "m", "y")

    #создание карты расстояний для ускорения работы алгоритма
    distmap = simple_distmap(dataset, distance_func)

    #список точек, расстояние от которых до точки p меньше, чем eps
    def dense_points(point):
        point_index = np.where(dataset == point)[0][0]
        points_list = []
        for i in range(distmap.shape[0]):
            if distmap[point_index, i] < eps:
                points_list.append(dataset[i])
        return points_list

    def create_cluster(point, neighbours):
        #создание нового списка точек для созданного кластера
        clusters[C] = []
        #добавление данной точки в созданный кластер
        clusters[C].append(tuple(point))
        #пометра точки как кластеризованной
        clustered_points.add(tuple(point))


        #просмотр точек в окрестности данной точки (соседей)
        while neighbours:
            #point_n - сосед точки (point)
            point_n = tuple(neighbours.pop())
            #если point_n не была посещена ранее
            if point_n not in visited_points:
                #отметить point_n как посещённую
                visited_points.add(point_n)
                # соседи point_n
                if visualize:
                    plt.scatter(point_n[0], point_n[1], color = colors[int(C % len(colors))])
                    plt.draw()
                    plt.gcf().canvas.flush_events()
                point_neighbours = dense_points(point_n)
                if len(point_neighbours) > min_neighbours:
                    neighbours.extend(point_neighbours)

            if point_n not in clustered_points:
                clustered_points.add(point_n)
                clusters[C].append(point_n)
                if point_n in clusters[NOISE]:
                    clusters[NOISE].remove(point_n)
  

    for point in dataset:
        #просмотр только непосещённых точек
        if tuple(point) in visited_points:
            continue
        #классификация точки как посещённой
        visited_points.add(tuple(point))
        #поиск ближайших соседей точки
        neighbours = dense_points(point)
        #если соседей меньше, чем минимальное количество соседей
        if len(neighbours) < min_neighbours:
            #то точка считается потенциальным шумом
            clusters[NOISE].append(tuple(point)) 
        #иначе 
        else:
            #добавление нового кластера
            C += 1
            #расширение нового кластера
            create_cluster(point, neighbours)

    if visualize:
        plt.ioff()
    return clusters

#для одинаковой генерации набора данных
np.random.seed(10)
#генерация набора данных
dataset = datasets.make_circles(n_samples=300, factor=0.5, noise=0.04)
#добавление выбросов
dataset = (np.vstack((dataset[0], np.array([[-1., -1.], [1., -1.], [1., 1.], [0, 0], [0.1, 0.1], [-0.75, 0]]))), np.hstack((dataset[1], np.array([2, 2, 2, 2, 2, 2]))))

clusters_dict = DBSCAN(0.2, 4, euclid, dataset[0], False)
clusters_labels = label_clusters(dataset, clusters_dict)
print(clusters_labels)

'''
#запуск алгоритма с различными настроечными значениями
parameters = [np.linspace(0.1, 0.5, 9), np.arange(1, 10)]
print(parameters)
accuracies = []
for val_0 in parameters[0]:
    accuracies.append([])
    for val_1 in parameters[1]:
        clusters_dict = DBSCAN(val_0, val_1, euclid, dataset[0], False)

        #трансформация разметки кластеров и оценка точности кластеризации
        clusters_labels = label_clusters(dataset, clusters_dict)
        accuracies[-1].append(round(v_measure_score(dataset[1], clusters_labels), 2))


print(accuracies)
fig, ax = plt.subplots()
sns.heatmap(accuracies, cmap = sns.cubehelix_palette(as_cmap=True), annot = True)
def my_formatter_y(y, pos):
    return round(y/20 + 0.07, 2)

def my_formatter_x(x, pos):
    return x + 0.5

ax.yaxis.set_major_formatter(FuncFormatter(my_formatter_y))
ax.xaxis.set_major_formatter(FuncFormatter(my_formatter_x))

ax.set_xlabel('min_neighbours')
ax.set_ylabel('min_distance')

plt.show()
'''


#запуск алгоритма для различных значений размера выборки
'''
sample_sizes = np.arange(10, 301, 10)
print(sample_sizes)
accuracies = []
for i in sample_sizes:
    sample_ind = np.random.choice(dataset[0].shape[0], i, replace = False)
    cur_sample = dataset[0][sample_ind]
    clusters_dict = DBSCAN(0.2, 4, euclid, cur_sample, False)
    #трансформация разметки кластеров и оценка точности кластеризации
    clusters_labels = label_clusters([cur_sample, dataset[1][sample_ind]], clusters_dict)
    accuracies.append(round(v_measure_score(dataset[1][sample_ind], clusters_labels), 2))

fig, ax = plt.subplots()
plt.plot(sample_sizes, accuracies)
ax.set_xlabel("Sample size")
ax.set_ylabel("v measure score")
plt.show()
print(accuracies)
'''