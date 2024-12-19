import random
from tkinter import *

import numpy as np
from sklearn import preprocessing

# Задание №1. Предобработка данных. Автор: Литовченко. М. Д.

# Входные данные
# Задание размеров массивов
shape_1 = (4, 3)  # Для первого набора
shape_2 = (4, 3)  # Для второго набора

# Генерация случайных значений
input_data_1 = np.random.uniform(-10, 10, size=shape_1)
input_data_2 = np.random.uniform(-100, 100, size=shape_2)

# Набор номер 3 в виде ключ значение
input_data_3 = [['Мышь', 1], ['Кот', 3], ['Мышь', 2], ['Мышь', 1], ['Кот', 3]]


# Обработка
def button_clicked_1():
    output_text.delete(1.0, END)  # Очистка текстового поля перед выводом
    output_text.insert(END, 'Входные данные:\n')
    output_text.insert(END, str(input_data_1) + '\n')

    output_text.insert(END, 'Стандартизованные данные набор_1:\n')
    input_data_1_scaled = preprocessing.scale(input_data_1)
    output_text.insert(END, str(input_data_1_scaled) + '\n')

    scaler = preprocessing.StandardScaler().fit(input_data_1)
    output_text.insert(END, 'Стандартизованные данные набор_2 (StandardScaler):\n')
    output_text.insert(END, str(scaler.transform(input_data_2)) + '\n')

    output_text.insert(END, 'Стандартизованные данные набор_1 (MinMaxScaler):\n')
    min_max_scaler = preprocessing.MinMaxScaler()
    input_data_1_minmax = min_max_scaler.fit_transform(input_data_1)
    output_text.insert(END, str(input_data_1_minmax) + '\n')

    output_text.insert(END, 'Стандартизованные данные набор_1 (MaxAbsScaler):\n')
    max_abs_scaler = preprocessing.MaxAbsScaler()
    input_data_1_maxabs = max_abs_scaler.fit_transform(input_data_1)
    output_text.insert(END, str(input_data_1_maxabs) + '\n')

    output_text.insert(END, 'Нормализованные данные набор_1 (L1-нормализация):\n')
    data_normalized_l1 = preprocessing.normalize(input_data_1, norm='l1')
    output_text.insert(END, str(data_normalized_l1) + '\n')

    output_text.insert(END, 'Нормализованные данные набор_1 (L2-нормализация):\n')
    data_normalized_l2 = preprocessing.normalize(input_data_1, norm='l2')
    output_text.insert(END, str(data_normalized_l2) + '\n')

    output_text.insert(END, 'Кодированные данные набор_1 (OrdinalEncoder):\n')
    enc = preprocessing.OrdinalEncoder()
    enc.fit(input_data_3)
    output_text.insert(END, str(enc.categories_) + '\n')
    output_text.insert(END, str(enc.transform(input_data_3)) + '\n')


def button_clicked_2():
    output_text.delete(1.0, END)  # Очистка текстового поля перед выводом

    for _ in range(0, 5):
        random_n_bins = random.randint(2, 10)
        output_text.insert(END, 'Дискретизация данных набор_2:\n')
        output_text.insert(END, f'Значение random_n_bins: {random_n_bins}\n')
        diskr = preprocessing.KBinsDiscretizer(n_bins=random_n_bins, encode='ordinal', strategy='uniform')
        diskr.fit(input_data_2)
        input_data_1_diskr = diskr.transform(input_data_2)
        output_text.insert(END, str(input_data_1_diskr) + '\n')


def button_clicked_3():
    output_text.delete(1.0, END)  # Очистка текстового поля перед выводом

    for _ in range(0, 3):
        random_threshold = round(random.uniform(1, 10), 1)
        output_text.insert(END, 'Бизаризация данных набор_1:\n')
        output_text.insert(END, f'Значение random_threshold: {random_threshold}\n')
        input_data_1_binarized = preprocessing.Binarizer(threshold=random_threshold).transform(input_data_1)
        output_text.insert(END, str(input_data_1_binarized) + '\n')


if __name__ == "__main__":
    root = Tk()
    root.title("Предобработка данных")

    # Создание кнопок
    button1 = Button(root, text="Предобработка", command=button_clicked_1)
    button1.pack()

    button2 = Button(root, text="Дискретизация", command=button_clicked_2)
    button2.pack()

    button3 = Button(root, text="Бинаризация", command=button_clicked_3)
    button3.pack()

    # Текстовое поле для вывода результатов
    output_text = Text(root, wrap='word')
    output_text.pack(fill='both', expand=True)

    root.mainloop()
