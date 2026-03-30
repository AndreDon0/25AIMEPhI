# МИФИ ML 2025/2026

Данные материалы подготовлены для годового курса машинного обучения в НИЯУ МИФИ. \
Цель курса - познакомить с инструментами для работы с машимнным обучением, нейронными сетями, а также с самими моделями, применительно к реальным задачам. 

Авторы курса: Егоров Алексей и Егорова Аделя

## Quick start (using `uv`)

1. Install `uv` (macOS example):
   ```bash
   brew install uv
   ```
2. Create/sync the environment from `pyproject.toml`:
   ```bash
   uv sync
   ```
3. (Optional) Register the notebook kernel:
   ```bash
   uv run python -m ipykernel install --user --name 25aimephi --display-name "25aimephi (Python)"
   ```
4. Open any notebook (for example `Homework_11_neuro_regression/main.ipynb`) and select the kernel `25aimephi (Python)`.

Sanity check:
```bash
uv run python -c "import numpy, torch; print('env ok')"
```

# Модуль №1 Программирование на python для машнного обучения

Формы контроля: коллоквиум по итогам модуля.

[0] [Установочная презентация](https://github.com/ShadarRim/25AIMEPhI/blob/main/00_%D0%9F%D1%80%D0%B5%D0%B7%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8F%20%D0%BE%20ML.pptx) \
[1] [Основы Python 1](https://github.com/ShadarRim/25AIMEPhI/blob/main/01_%D0%9E%D1%81%D0%BD%D0%BE%D0%B2%D1%8B_Python_1.ipynb) \
[2] [Основы Python 2](https://github.com/ShadarRim/25AIMEPhI/blob/main/02_%D0%9E%D1%81%D0%BD%D0%BE%D0%B2%D1%8B_Python_2.ipynb) \
[3] [Numpy](https://github.com/ShadarRim/25AIMEPhI/blob/main/03_Numpy.ipynb) \
[4] [Pandas 1](https://github.com/ShadarRim/25AIMEPhI/blob/main/04_Pandas.ipynb) \
[5] [Pandas 2](https://github.com/ShadarRim/25AIMEPhI/blob/main/05_Pandas%202.ipynb) \
[6] [Визуализация](https://github.com/ShadarRim/25AIMEPhI/blob/main/06_%D0%92%D0%B8%D0%B7%D1%83%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F_%2B_EDA.ipynb) \
[7] [Элементы ТВиМС](https://github.com/ShadarRim/25AIMEPhI/blob/main/07_%D0%97%D0%BD%D0%B0%D0%BA%D0%BE%D0%BC%D1%81%D1%82%D0%B2%D0%BE%20%D1%81%20%D1%8D%D0%BB%D0%B5%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D0%BC%D0%B8%20%D0%A2%D0%92%D0%B8%D0%9C%D0%A1.ipynb) \
[8] [Точечные оценки](https://github.com/ShadarRim/25AIMEPhI/blob/main/08_%D0%A2%D0%BE%D1%87%D0%B5%D1%87%D0%BD%D1%8B%D0%B5%20%D0%BE%D1%86%D0%B5%D0%BD%D0%BA%D0%B8%20%D0%A2%D0%92%D0%B8%D0%9C%D0%A1.ipynb) \
[9] [Гипотезы и ошибки](https://github.com/ShadarRim/25AIMEPhI/blob/main/09_%D0%93%D0%B8%D0%BF%D0%BE%D1%82%D0%B5%D0%B7%D1%8B%20%D0%B8%20%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B8.ipynb) \
[10] [Проверки](https://github.com/ShadarRim/25AIMEPhI/blob/main/10_%D0%9F%D1%80%D0%BE%D0%B2%D0%B5%D1%80%D0%BA%D0%B8.ipynb) \

# Модуль №2 Классическое машинное обучение

[11] [Введение в ML](https://github.com/ShadarRim/25AIMEPhI/blob/main/11_ML_intruduction.ipynb)\
[12] [Градиентный спуск](https://github.com/ShadarRim/25AIMEPhI/blob/main/12_%D0%93%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D0%B9%20%D1%81%D0%BF%D1%83%D1%81%D0%BA.ipynb)\
[13] [Линейная регрессия с нуля](https://github.com/ShadarRim/25AIMEPhI/blob/main/13_LinReg_scratch.ipynb)\
[14] [Линейная регрессия sklearn](https://github.com/ShadarRim/25AIMEPhI/blob/main/14_%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F_%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F_%D0%BF%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D0%BA%D0%B0_1.ipynb)\
[15] [Решение задач регрессии, обработка данных](https://github.com/ShadarRim/25AIMEPhI/blob/main/14_%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F_%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F_%D0%BF%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D0%BA%D0%B0_1.ipynb)\
[16] [Логистическая регрессия](https://github.com/ShadarRim/25AIMEPhI/blob/main/16_%D0%9B%D0%BE%D0%B3%D0%B8%D1%81%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F.ipynb)\
[17] [Оценка качества классификации](https://github.com/ShadarRim/25AIMEPhI/blob/main/17_%D0%9E%D1%86%D0%B5%D0%BD%D0%BA%D0%B0_%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%B0_%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8.ipynb)\
[18] [Работа с категориальными признаками](https://github.com/ShadarRim/25AIMEPhI/blob/main/18_%D0%A0%D0%B0%D0%B1%D0%BE%D1%82%D0%B0_%D1%81_%D0%BA%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%BC%D0%B8_%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%BC%D0%B8_1.ipynb)\
[19] [Дерево решений](https://github.com/ShadarRim/25AIMEPhI/blob/main/19_%D0%94%D0%B5%D1%80%D0%B5%D0%B2%D0%BE_%D1%80%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D1%8F.ipynb)\
[20] [Случайный лес](https://github.com/ShadarRim/25AIMEPhI/blob/main/20_%D0%A1%D0%BB%D1%83%D1%87%D0%B0%D0%B9%D0%BD%D1%8B%D0%B9_%D0%BB%D0%B5%D1%81.ipynb)\
[21] [Сравнение леса и дерева](https://github.com/ShadarRim/25AIMEPhI/blob/main/21_%D0%A1%D1%80%D0%B0%D0%B2%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_%D0%BB%D0%B5%D1%81%D0%B0_%D0%B8_%D0%B4%D0%B5%D1%80%D0%B5%D0%B2%D0%B0_%2B_%D0%BF%D0%BE%D0%B4%D0%B1%D0%BE%D1%80_%D0%BF%D0%B0%D1%80%D0%B0%D0%BC%D0%B5%D1%82%D1%80%D0%BE%D0%B2.ipynb)\
[22] [Бустинг](https://github.com/ShadarRim/25AIMEPhI/blob/main/22_%D0%91%D1%83%D1%81%D1%82%D0%B8%D0%BD%D0%B3.ipynb)\
[23] [Балансировка классов](https://github.com/ShadarRim/25AIMEPhI/blob/main/23_%D0%91%D0%B0%D0%BB%D0%B0%D0%BD%D1%81%D0%B8%D1%80%D0%BE%D0%B2%D0%BA%D0%B0%20%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%BE%D0%B2.ipynb)\

# Модуль №3 Нейронные сети 

[24] [О нейронных сетях](https://github.com/ShadarRim/25AIMEPhI/blob/main/24_%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D1%8B%D0%B5%20%D1%81%D0%B5%D1%82%D0%B8.pptx)\
[25] [Tensor Torch](https://github.com/ShadarRim/25AIMEPhI/blob/main/25_%D0%A2%D0%B5%D0%BD%D0%B7%D0%BE%D1%80%D1%8B_Torch.ipynb)\
[26] [Tensor Torch вручную](https://github.com/ShadarRim/25AIMEPhI/blob/main/26_Tensor%20with%20broadcast.ipynb) \
[27] [Модицифкация градиентного спуска](https://github.com/ShadarRim/25AIMEPhI/blob/main/27_%D0%9C%D0%BE%D0%B4%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8_%D0%B3%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D0%BE%D0%B3%D0%BE_%D1%81%D0%BF%D1%83%D1%81%D0%BA%D0%B0.ipynb) \
[28] [Нейронная сеть без градиентого спуска](https://github.com/ShadarRim/25AIMEPhI/blob/main/28_%D0%9E%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_%D0%B1%D0%B5%D0%B7_%D0%B3%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%B0.ipynb) \
[29] [НС для задачи классификации](https://github.com/ShadarRim/25AIMEPhI/blob/main/30_%D0%9D%D0%A1_%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D1%8F.ipynb) \
[30] [НС для задачи регрессии](https://github.com/ShadarRim/25AIMEPhI/blob/main/31_%D0%9D%D0%A1_%D0%A0%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F.ipynb) \
[31] [CNN](https://github.com/ShadarRim/25AIMEPhI/blob/main/32_%D0%A1%D0%B2%D1%91%D1%80%D1%82%D0%BE%D1%87%D0%BD%D1%8B%D0%B5%20%D1%81%D0%B5%D1%82%D0%B8.pptx) \
[32] [MNIST](https://github.com/ShadarRim/25AIMEPhI/blob/main/33_MNIST_CNN.ipynb) \
[33] [Работа с изображениями](https://github.com/ShadarRim/25AIMEPhI/blob/main/35_%D0%A3%D0%BB%D1%83%D1%87%D1%88%D0%B5%D0%BD%D0%B8%D0%B5_%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%B0_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D1%81%D0%B5%D1%82%D0%B5%D0%B9.ipynb) \
[34] [Улучшение качества обучения нейронной сети](https://colab.research.google.com/drive/1vINdRci8tc760Q4mzv-jIQJG5RQn6HuT?usp=sharing)\
[35] [Об аугментации](https://github.com/ShadarRim/25AIMEPhI/blob/main/36_%D0%9D%D0%B5%D0%BC%D0%BD%D0%BE%D0%B3%D0%BE_%D0%BF%D1%80%D0%BE_%D0%B0%D1%83%D0%B3%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8E.ipynb)\
[36] [Задачи компьютерного зрения](https://github.com/ShadarRim/25AIMEPhI/blob/main/37_%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B8_%D0%9A%D0%97.pptx)\
[37] [Рещение задачи сегментации "из коробки"](https://github.com/ShadarRim/25AIMEPhI/blob/main/38_%D0%A1%D0%B5%D0%B3%D0%B5%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8F_Pytorch_box.ipynb)

# Самостоятельная работа на паре
[1] [Python](https://contest.yandex.ru/contest/81119/enter) \
[2] [Numpy](https://contest.yandex.ru/contest/81472/enter)

# Домашнее задание осеннего семестра

Домашние задания бывают обязательные и необязательные. Все ДЗ, что с номерами - обязательные. 

[0] [Контест на Python](https://contest.yandex.ru/contest/80799/enter/?retPage=) \
[1] [Контест на NumPy](https://contest.yandex.ru/contest/81120/enter) \
[2] [Контест на Pandas](https://contest.yandex.ru/contest/81473/enter) \
[3] Обязательное ДЗ №1. Подготовить для самостоятельно выбранного набора данных анализ различных гипотез, оформелнный, как в виде написанного кода, так и в виде отчёта с аналитической человекопонятной частью. \
[4] Обязательное ДЗ №2. Написать линейную регрессию с нуля самостоятельно. \
[5] Обязательное ДЗ №3. Решить задачу регрессии на kaggle в соответствии с вариантом. \
[6] Обязательное ДЗ №4. Написать логистическую регрессию с нуля самостоятельно. Создать для неё класс. \
[7] Обязательное ДЗ №5. Решить задачу классификации на kaggle в соответствии с вариантом. \
[8] Обязательное ДЗ №6. Реализовать дерево решения самому. \
[9] Обязательное ДЗ №7. Реализовать случайный лес самому. \
[10] Обязательное ДЗ №8. Решить задачу кластеризации на kaggle.

# Домашенее задание весеннего семестра
[11] [Контест на Torch](https://contest.yandex.ru/contest/90000/standings/) \
[12] Обязательное ДЗ№1. Разъяснить каждую строчку Tensor написаного с нуля. \
[13] Обязательное ДЗ№2. Решить задачу регресии из прошлого семестра с использованием полносвязной НС. \
[14] Обязательное ДЗ№3. Решить задачу классификации из прошлого семестра с использованием полносвязной НС. \
[15] [Обязательное ДЗ№4. Решить задачу классификации бабочек (late submission).](https://www.kaggle.com/competitions/classification-of-butterflies) \
[16] Обязательное ДЗ№5. Повысить точность решения MNIST до 98%. \
[17] [Обязательное ДЗ№6. Решить задачу с точностью не менее 95%.](https://www.kaggle.com/c/platesv2) \
[18] [Обязательное ДЗ№7. Решить задачу на нахождение объектов](https://www.kaggle.com/competitions/find-the-seagulls/overview) \
[19] [Обязательное ДЗ№8. Решить задачу сегментации кружочков.](https://github.com/ShadarRim/25AIMEPhI/blob/main/a07_%D0%A1%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BA%D1%80%D1%83%D0%B6%D0%BE%D1%87%D0%BA%D0%BE%D0%B2.ipynb) \
[20] [Обязательное ДЗ№9. Решить задачу классификации объектов на видео](https://www.kaggle.com/competitions/what-on-the-video/overview) \
[21] [Обязательное ДЗ№10. Сгенрировать с помощью GAN или диффузионной модели собственное сэлфи] \

# Дополнительные материалы
[0] [Пара фишек по работе с Python](https://github.com/ShadarRim/25AIMEPhI/blob/main/a00_%D0%9F%D0%B0%D1%80%D0%B0_%D1%84%D0%B8%D1%88%D0%B5%D0%BA%20Python.ipynb)\
[1] [Работа с файлами в Python](https://github.com/ShadarRim/25AIMEPhI/blob/main/a01_%D0%A0%D0%B0%D0%B1%D0%BE%D1%82%D0%B0_%D1%81_%D1%84%D0%B0%D0%B9%D0%BB%D0%B0%D0%BC%D0%B8.ipynb) \
[2] [Math](https://github.com/ShadarRim/25AIMEPhI/blob/main/a02_%D0%9C%D0%BE%D0%B4%D1%83%D0%BB%D1%8C_math.ipynb) \
[3] [Что делает этот код? Numpy](https://github.com/ShadarRim/25AIMEPhI/blob/main/a03_%D0%A7%D1%82%D0%BE_%D0%B4%D0%B5%D0%BB%D0%B0%D0%B5%D1%82_%D1%8D%D1%82%D0%BE%D1%82_%D0%BA%D0%BE%D0%B4.ipynb)

# Олимпиады и конкурсы
Участие в олимпиадах и конкурсах позволяет повысить итоговый балл в семестре. 

Выход в финал добавляет баллы.

[0] ["Я-профессионал"](https://yandex.ru/profi/) \
[1] [Полное закрытие курса за медаль в любом соревновании на kaggle](kaggle.com) \
[2] [Внутренний хакатон НИЯУ МИФИ](https://t.me/mephihack/231)