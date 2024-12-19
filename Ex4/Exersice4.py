from tkinter import Tk, Button

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans


def run_kmeans():
    X = np.loadtxt('k_means.txt', delimiter=',')

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.title('Входные данные')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    scores = []
    arange_end = 20
    arange_start = 2
    print(f"arange_start{arange_start} "
          f"arange_end{arange_end} ")
    values = np.arange(arange_start, arange_end)

    num_clusters_t = 0

    for num_clusters in values:
        # Обучение модели кластеризации КМеаns
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
        kmeans.fit(X)
        # Получить силуэтную оценку
        score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean', sample_size=len(X))
        print("\nКоличество кластеров =", num_clusters)
        print("Силуэтная оценка =", score)
        scores.append(score)

        num_clusters_temp = np.argmax(scores) + values[0]

        if num_clusters_temp > num_clusters_t:
            optimap_kmeans = kmeans
            num_clusters_t = num_clusters_temp

    plt.figure()
    plt.bar(values, scores, width=0.7, color='black', align='center')
    plt.title('Силуэтная оценка числа кластеров')

    num_clusters = np.argmax(scores) + values[0]
    print('\nОптимальное количество кластеров =', num_clusters)
    plt.show()

    # Визуализация границ кластеров
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=optimap_kmeans.labels_, cmap='viridis')
    centers = optimap_kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = optimap_kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, n_levels=num_clusters, cmap='viridis', alpha=0.4)
    plt.title(f'Границы кластеров (n_clusters=4)')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


# Создание главного окна
root = Tk()
root.title("Кластеризация")

# Создание кнопки для запуска классификатора
run_button = Button(root, text="Кластеризация", command=run_kmeans)
run_button.pack(pady=20)

# Запуск основного цикла приложения
root.mainloop()
