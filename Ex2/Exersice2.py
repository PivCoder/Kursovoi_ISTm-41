from tkinter import *

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


# Задание №2. Изучение возможностей методов обучения с учителем. Автор: Литовченко. М. Д.
def run_SVM_vectors():
    input_file = 'SVM.txt'
    X = []
    y = []
    count_class1 = 0
    count_class2 = 0
    max_datapoints = 500

    with open(input_file, 'r') as f:
        for line in f.readlines():
            if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                break
            if '?' in line:
                continue

            data = line[:-1].split(', ')

            if data[-1] == '<=50K' and count_class1 < max_datapoints:
                X.append(data)
            count_class1 += 1
            if data[-1] == '>50K' and count_class2 < max_datapoints:
                X.append(data)
            count_class2 += 1

    X = np.array(X)

    label_encoder = []
    X_encoded = np.empty(X.shape)
    for i, item in enumerate(X[0]):
        if item.isdigit():
            X_encoded[:, i] = X[:, i]
        else:
            label_encoder.append(preprocessing.LabelEncoder())
            X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

    X = X_encoded[:, :-1].astype(int)
    y = X_encoded[:, -1].astype(int)

    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
    classifier.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=5)
    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)

    output_text.delete("1.0", END)

    print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")
    num_folds = 3
    accuracy_values = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
    print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")
    precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
    print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")
    recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
    print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")
    f1_values = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
    print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")

    output_text.insert(END, "Результаты работы машины на опорных векторах:\n\n")
    output_text.insert(END, "| Метрика | Значение |\n")
    output_text.insert(END, "| --- | --- |\n")
    output_text.insert(END, f"| Precision | {precision_values} |\n")
    output_text.insert(END, f"| Precision | {round(100 * precision_values.mean(), 2)} |\n")
    output_text.insert(END, f"| Recall | {recall_values} |\n")
    output_text.insert(END, f"| Recall | {round(100 * recall_values.mean(), 2)} |\n")
    output_text.insert(END, f"| F1-score | {f1_values} |\n")
    output_text.insert(END, f"| F1-score | {round(100 * f1.mean(), 2)} |\n")
    output_text.insert(END, f"| Accuracy | {accuracy_values} |\n")
    output_text.insert(END, f"| Accuracy | {round(100 * accuracy_values.mean(), 2)} |\n")

    input_data_mass = [
        ['27', 'Self-emp-not-inc', '154789', 'Bachelors', '10', 'Married-civ-spouse', 'Tech-support', 'Husband',
         'White', 'Male', '0', '0', '10', 'Dominican-Republic'],
        ['38', 'Local-gov', '444333', 'HS-grad', '11', 'Married-civ-spouse', 'Adm-clerical', 'Husband',
         'White', 'Male', '0', '0', '50', 'United-States'],
        ['58', 'Private', '128965', 'Assoc-acdm', '9', 'Separated', 'Prof-specialty', 'Not-in-family',
         'Black', 'Male', '0', '0', '40', 'Dominican-Republic']]

    for input_data in input_data_mass:
        input_data_encoded = [-1] * len(input_data)
        count = 0
        for i, item in enumerate(input_data):
            if item.isdigit():
                input_data_encoded[i] = int(input_data[i])
            else:
                input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
                count += 1

        input_data_encoded = np.array([input_data_encoded])
        predicted_class = classifier.predict(input_data_encoded)
        print(label_encoder[-1].inverse_transform(predicted_class)[0])
        output_text.insert(END, f"| Result | {label_encoder[-1].inverse_transform(predicted_class)[0]} |\n")


# Функция для выполнения наивного байесовского классификатора
def run_naive_bayes():
    # Загрузка данных из файла
    input_file = 'baes.txt'
    data = np.loadtxt(input_file, delimiter=',')
    print(data)
    X, y = data[:, :-1], data[:, -1]
    classifier = GaussianNB()
    classifier.fit(X, y)

    output_text.delete("1.0", END)

    y_pred = classifier.predict(X)
    accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
    print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")
    output_text.insert(END, f"| Accuracy of Naive Bayes classifier | {accuracy} |\n")

    num_folds = 3
    accuracy_values = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
    print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")
    precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
    print("Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")
    recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
    print("Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")
    f1_values = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
    print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")

    # Вывод результатов в текстовое поле

    output_text.insert(END, "Результаты наивного байесовского классификатора:\n\n")
    output_text.insert(END, "| Метрика | Значение |\n")
    output_text.insert(END, "| --- | --- |\n")
    output_text.insert(END, f"| Precision | {precision_values} |\n")
    output_text.insert(END, f"| Recall | {recall_values} |\n")
    output_text.insert(END, f"| F1-score | {f1_values} |\n")
    output_text.insert(END, f"| Accuracy | {accuracy_values} |\n")


# Создание главного окна
root = Tk()
root.title("Наивный Байесовский Классификатор")

# Создание кнопки для запуска классификатора
run_button = Button(root, text="Запустить классификатор", command=run_naive_bayes)
run_button.pack(pady=20)

run_button = Button(root, text="Запустить метод опорных векторов", command=run_SVM_vectors)
run_button.pack(pady=20)

# Текстовое поле для вывода результатов
output_text = Text(root, wrap='word')
output_text.pack(fill='both', expand=True)

# Запуск основного цикла приложения
root.mainloop()
