from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

score = 0

def learn():
    # Загружаем датасет
    df = pd.read_csv("./CarPrice_Assignment.csv")
    # Дропаем столбец ID
    df.drop(columns="car_ID", inplace=True)

    # Меняем тип данных из объекта в численное значение
    object_data = df.select_dtypes(include='object')
    num_data = df.select_dtypes(exclude='object')
    enc = LabelEncoder()
    for i in range(0, object_data.shape[1]):
        object_data.iloc[:, i] = enc.fit_transform(object_data.iloc[:, i])

    #Объединяем данные
    full_data = pd.concat([num_data, object_data], axis=1)

    #Посмотрим на корреляции в данных
    full_data.corr()["price"].sort_values()

    #Выбираем колонки с самой большой корреляцией
    data1 = full_data[
        [
            "highwaympg", "citympg", "CarName", 'enginelocation', 'fuelsystem',
            'boreratio', 'wheelbase', 'drivewheel', 'carlength', 'carwidth',
            'horsepower', 'curbweight', 'enginesize', 'price'
        ]
    ]
    # Разделяем признаки
    X = data1.drop(columns="price")
    y = data1["price"]

    # Разделяем на тестовую и обучающую выборку
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70)

    # Выбираем различные алгоритмы обучения
    LR = LinearRegression()
    DTR = DecisionTreeRegressor()
    RFR = RandomForestRegressor()
    KNR = KNeighborsRegressor()

    # Вычислим точность каждого из алгоритмов
    model = [LR, DTR, RFR, KNR]
    d = {}
    for i in model:
        i.fit(X_train, y_train)
        ypred = i.predict(X_test)
        print(i, ":", r2_score(y_test, ypred) * 100)
        d.update({str(i): i.score(X_test, y_test) * 100})

    # Построим график точности для каждого из алгоритмов

    plt.figure(figsize=(30, 15))
    plt.title("Algorithm vs Accuracy")
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy")
    plt.plot(d.keys(), d.values(), marker='o', color='red')
    plt.show()

    # Standardscaler
    Sc_data = data1.copy()

    # Проведем те же манипуляции с данными
    A = Sc_data.drop(columns="price")
    b = Sc_data["price"]

    # Скейлим данные
    Scaler = StandardScaler()
    A = Scaler.fit_transform(A)

    # Снова разделим выборки
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(A, b, train_size=0.7)


    # Измерим точность
    model = [LR, DTR, RFR, KNR]
    q = {}
    for i in model:
        i.fit(X_train_s, y_train_s)
        ypred_s = i.predict(X_test_s)
        print(i, ":", r2_score(y_test_s, ypred_s) * 100)
        q.update({str(i): i.score(X_test_s, y_test_s) * 100})

    #Выше всего точность в RFR, так что используем его для обучения модели
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    global score
    score = model.score(X_test, y_test)

    # Функция возвращает обученную модель
    return model

def data_to_DF(data):

    # Чтобы конвертировать JSON в DataFrame нам нужен пустой DataFrame. В нем указываем
    # только те признаки, которые мы выбрали для обучения модели
    columns = [
        'highwaympg', 'citympg', 'CarName', 'enginelocation', 'fuelsystem',
        'boreratio', 'wheelbase', 'drivewheel', 'carlength', 'carwidth',
        'horsepower', 'curbweight', 'enginesize'
    ]
    df = pd.DataFrame(columns=columns)

    # С помощью функции loc мы перезаписываем элементы dataFrame, в данном случае нам нужен [0] элемент
    df.loc[0, 'highwaympg'] = data.get("highwaympg")
    df.loc[0, 'citympg'] = data.get("citympg")
    df.loc[0, 'CarName'] = data.get("carName")
    df.loc[0, 'enginelocation'] = data.get("enginelocation")
    df.loc[0, 'fuelsystem'] = data.get("fuelsystem")
    df.loc[0, 'boreratio'] = data.get("boreratio")
    df.loc[0, 'wheelbase'] = data.get("wheelbase")
    df.loc[0, 'drivewheel'] = data.get("drivewheel")
    df.loc[0, 'carlength'] = data.get("carlength")
    df.loc[0, 'carwidth'] = data.get("carwidth")
    df.loc[0, 'horsepower'] = data.get("horsepower")
    df.loc[0, 'curbweight'] = data.get("curbweight")
    df.loc[0, 'enginesize'] = data.get("enginesize")

    # Функция возвращает заполненный DataFrame, который может использоваться для предсказания
    return df

app = Flask(__name__)

# разрешаем CORS для всех доменов
CORS(app)

# API-метод для обработки POST-запроса
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Получаем данные из POST-запроса
    data = request.get_json()

    print('Что пришло: ', data)

    # Проверяем, что данные были переданы
    if data is None:
        return jsonify({'error': 'no data provided'})

    df = data_to_DF(data)

    print('Датафрейм: ', df)

    # Запускаем обучение модели при старте программы
    model = learn()

    # Возвращаем результаты обработки запроса в формате JSON
    return jsonify(model.predict(df)[0], score)

if __name__ == '__main__':
    # Запускаем сервер на локальном хосте и порту 8000
    app.run(host='localhost', port=8000, debug=True)

