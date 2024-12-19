import random
from tkinter import *

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, median_absolute_error, \
    r2_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier


def gnerate():
    # Задаем количество записей
    num_records = 141

    # Генерируем случайные данные
    X1 = np.round(np.random.uniform(1, 10, size=num_records), 2)
    X2 = np.round(np.random.uniform(1, 10, size=num_records), 2)
    X3 = np.round(np.random.uniform(0, 1, size=num_records), 0)

    # Сохраняем данные в файл
    with open('data_decision_trees.txt', 'a') as f:
        for i in range(num_records):
            f.write(f"{X1[i]},{X2[i]},{X3[i]}\n")

def run_liner_regression():
    input_file = 'data_singlevar_regr.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    num_training = int(0.8 * len(X))
    num_test = len(X) - num_training
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, y_train)
    y_test_pred = regressor.predict(X_test)

    print("Результат работы линейной регрессии")
    print("Mean absolute error =", round(mean_absolute_error(y_test, y_test_pred), 2))
    print("Mean squared error =", round(mean_squared_error(y_test, y_test_pred), 2))
    print("Median absolute error =", round(median_absolute_error(y_test, y_test_pred), 2))
    print("Explain variance score =", round(explained_variance_score(y_test, y_test_pred), 2))
    print("R2 score =", round(r2_score(y_test, y_test_pred), 2))

    output_text.delete("1.0", END)
    output_text.insert(END, "Результаты работы линейной регрессии:\n\n")
    output_text.insert(END, "| Метрика | Значение |\n")
    output_text.insert(END, "| --- | --- |\n")
    output_text.insert(END, f"| Mean absolute error | {round(mean_absolute_error(y_test, y_test_pred), 2)} |\n")
    output_text.insert(END, f"| Mean squared error | {round(mean_squared_error(y_test, y_test_pred), 2)} |\n")
    output_text.insert(END, f"| Median absolute error | {round(median_absolute_error(y_test, y_test_pred), 2)} |\n")
    output_text.insert(END, f"| Explain variance score | {round(explained_variance_score(y_test, y_test_pred), 2)} |\n")
    output_text.insert(END, f"| R2 score | {round(r2_score(y_test, y_test_pred), 2)} |\n")


def run_polynomial_regression():
    # Загрузка данных из файла
    data = np.genfromtxt('data_multivar_regr.txt', delimiter=',')
    X = data[:, :-1]  # Признаки
    y = data[:, -1]  # Целевая переменная

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели многомерной линейной регрессии
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    # Оценка качества модели на тестовой выборке
    y_pred = model.predict(X_test)
    linear_regressor = linear_model.LinearRegression()
    linear_regressor.fit(X_train, y_train)

    print("Метрики качества модели:")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Median Absolute Error: {median_absolute_error(y_test, y_pred):.2f}")
    print(f"Explained Variance Score: {explained_variance_score(y_test, y_pred):.2f}")
    print(f"R2 Score: {explained_variance_score(y_test, y_pred):.2f}")

    polynomial = PolynomialFeatures(degree=10)
    X_train_transformed = polynomial.fit_transform(X_train)
    datapoint = [[[0.38, 0.41, 4.58]], [[0.20, 2.71, 2.58]], [[3.75, 4.85, 2.05]], [[3.88, 4.53, 0.06]],
                 [[4.35, -1.93, 5.38]]]

    output_text.delete("1.0", END)
    output_text.insert(END, "Результаты работы многомерной регрессии:\n\n")
    output_text.insert(END, "| Метрика | Значение |\n")
    output_text.insert(END, "| --- | --- |\n")
    output_text.insert(END, f"| Mean Absolute Error | {mean_absolute_error(y_test, y_pred):.2f} |\n")
    output_text.insert(END, f"| Mean Squared Error | {mean_squared_error(y_test, y_pred):.2f} |\n")
    output_text.insert(END, f"| Median Absolute Error | {median_absolute_error(y_test, y_pred):.2f} |\n")
    output_text.insert(END, f"| Explained Variance Score | {explained_variance_score(y_test, y_pred):.2f} |\n")
    output_text.insert(END, f"| R2 Score | {explained_variance_score(y_test, y_pred):.2f} |\n")
    output_text.insert(END, f"| R2 Score | {explained_variance_score(y_test, y_pred):.2f} |\n")

    for data in datapoint:
        poly_datapoint = polynomial.fit_transform(data)

        poly_linear_model = linear_model.LinearRegression()
        poly_linear_model.fit(X_train_transformed, y_train)
        print("\nЛинейная регрессия: \n", linear_regressor.predict(data))
        print("\nМногомерная регрессия: \n", poly_linear_model.predict(poly_datapoint))
        output_text.insert(END, f"| Линейная регрессия: | {linear_regressor.predict(data)} |\n")
        output_text.insert(END, f"| Многомерная регрессия: | {poly_linear_model.predict(poly_datapoint)} |\n")


def run_tree():
    input_file = 'data_decision_trees.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    test_size_coef = 0.54
    random_state_coef = 5
    param_random_state_coef = 1
    max_depth_coef = 4
    print(f"test_size_coef = {test_size_coef} "
          f"random_state_coef = {random_state_coef} "
          f"param_random_state_coef = {param_random_state_coef} "
          f"max_depth_coef = {max_depth_coef}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_coef,
                                                        random_state=random_state_coef)
    params = {'random_state': param_random_state_coef, 'max_depth': max_depth_coef}
    classifier = DecisionTreeClassifier(**params)
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)

    class_names = ['Class-0', 'Class-1']
    print("\n" + "#" * 40)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("#" * 40 + "\n")
    print("#" * 40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#" * 40 + "\n")

    output_text.delete("1.0", END)
    output_text.insert(END, "Результаты работы с деревом решений:\n\n")
    output_text.insert(END, "Тренировочный набор:\n\n")
    output_text.insert(END, f"{classification_report(y_train, classifier.predict(X_train), target_names=class_names)} |\n")
    output_text.insert(END, "Основной набор:\n\n")
    output_text.insert(END, f"{classification_report(y_test, y_test_pred, target_names=class_names)} |\n")


def run_random_forest():
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    test_size_coef = 0.64
    random_state_coef = 3
    n_estimators_coef = 55
    max_depth_coef = 8
    random_state_param_coef = 5
    print(f"test_size_coef = {test_size_coef} "
          f"random_state_coef = {random_state_coef} "
          f"n_estimators_coef = {n_estimators_coef} "
          f"max_depth_coef = {max_depth_coef} "
          f"random_state_param_coef = {random_state_param_coef}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_coef, random_state=random_state_coef)

    params = {'n_estimators': n_estimators_coef,
              'max_depth': max_depth_coef,
              'random_state': random_state_param_coef}

    classifier = RandomForestClassifier(**params)
    classifier.fit(X_train, y_train)

    y_test_pred = classifier.predict(X_test)

    class_names = ['Class-0', 'Class-1', 'Class-2']
    print("\n" + "#" * 40)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("#" * 40 + "\n")
    print("#" * 40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#" * 40 + "\n")

    output_text.delete("1.0", END)
    output_text.insert(END, "Результаты работы с случайным лесом:\n\n")
    output_text.insert(END, "Тренировочный набор:\n\n")
    output_text.insert(END,
                       f"{classification_report(y_train, classifier.predict(X_train), target_names=class_names)} |\n")
    output_text.insert(END, "Основной набор:\n\n")
    output_text.insert(END, f"{classification_report(y_test, y_test_pred, target_names=class_names)} |\n")


def run_predel_random_forest():
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    test_size_coef = 0.64
    random_state_coef = 3
    n_estimators_coef = 55
    max_depth_coef = 8
    random_state_param_coef = 5
    print(f"test_size_coef = {test_size_coef} "
          f"random_state_coef = {random_state_coef} "
          f"n_estimators_coef = {n_estimators_coef} "
          f"max_depth_coef = {max_depth_coef} "
          f"random_state_param_coef = {random_state_param_coef}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_coef, random_state=random_state_coef)

    params = {'n_estimators': n_estimators_coef,
              'max_depth': max_depth_coef,
              'random_state': random_state_param_coef}

    classifier = ExtraTreesClassifier(**params)
    classifier.fit(X_train, y_train)

    y_test_pred = classifier.predict(X_test)

    class_names = ['Class-0', 'Class-1', 'Class-2']
    print("\n" + "#" * 40)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("#" * 40 + "\n")
    print("#" * 40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#" * 40 + "\n")

    output_text.delete("1.0", END)
    output_text.insert(END, "Результаты работы с предельно случайным лесом:\n\n")
    output_text.insert(END, "Тренировочный набор:\n\n")
    output_text.insert(END,
                       f"{classification_report(y_train, classifier.predict(X_train), target_names=class_names)} |\n")
    output_text.insert(END, "Основной набор:\n\n")
    output_text.insert(END, f"{classification_report(y_test, y_test_pred, target_names=class_names)} |\n")

# Создание главного окна
root = Tk()
root.title("Регрессии, леса, деревья")

# Создание кнопки для запуска классификатора
run_button = Button(root, text="Линейная регрессия", command=run_liner_regression)
run_button.pack(pady=20)

run_button = Button(root, text="Многомерная регрессия", command=run_polynomial_regression)
run_button.pack(pady=20)

run_button = Button(root, text="Дерево", command=run_tree)
run_button.pack(pady=20)

run_button = Button(root, text="Случайный лес", command=run_random_forest)
run_button.pack(pady=20)

run_button = Button(root, text="Предельно случайный лес", command=run_predel_random_forest)
run_button.pack(pady=20)

# Текстовое поле для вывода результатов
output_text = Text(root, wrap='word')
output_text.pack(fill='both', expand=True)

# Запуск основного цикла приложения
root.mainloop()
