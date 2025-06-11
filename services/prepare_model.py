from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib

def model_predict(path_to_file):
    df = pd.read_excel(path_to_file)
    df = df.iloc[:-1].copy()

    cols_to_drop = [
        'Группа по кол-ву в общих продажах', 'ID',
        'Группа по продажам в общих продажах', 'Группа по доходу в общих продажах', 'Категория 1',
        'Категория 3', 'Категория 4', 'Поставщик', 'Оценка', 'Кол-во оценок', 'Актуальная цена из номенклатуры закупа',
        'Цена реализации в зале по меню', 'Выручка',
        'СрВзв размер уценки', 'СрВзв размер скидки по ПЛ', 'Себестоимость списаний',
        'Себестоимость продаж и списаний', 'Доход от продаж', 'Рентабельность',
        'Расчётная рентабельность', 'Наценка в стоимости по меню',
        'Наценка в фактической цене реализации',
        'Доля по кол-ву в общих продажах (%)',
        'Группа по кол-ву в общих продажах', 'Доля по кол-ву  (%)',
        'Группа по кол-ву', 'Доля по продажам в общих продажах (%)',
        'Группа по продажам в общих продажах', 'Доля по продажам  (%)',
        'Группа по продажам', 'Продажа по меню', 'Себестоимость продаж',
        'Доля по доходу в общих продажах (%)',
        'Группа по доходу в общих продажах', 'Доля по доходу от продаж  (%)',
        'Фудкост по стоимости в меню (%)'
    ]

    # prepare dataset
    df.drop(cols_to_drop, axis=1, inplace=True)
    df = df.dropna()
    df['Дата запуска'] = pd.to_datetime(df['Дата запуска'], format='%d.%m.%Y')
    df["CustomersInMonh"] = 0

    # endcoding
    label_encoder = LabelEncoder()
    df_dish = df['Блюдо'].copy()
    df['Блюдо'] = label_encoder.fit_transform(df['Блюдо'])

    # load models
    model = joblib.load('./resource/random_forest_model.pkl')
    scaler = joblib.load('./resource/scaler.pkl')

    new_df_scaled = scaler.transform(df.select_dtypes(include=[np.number]))

    redicted_value = model.predict(new_df_scaled)

    df["закупка"] = np.round(redicted_value)
    df['Блюдо'] = df_dish

    clean = [
        "Дата запуска", "Количество повторных продаж", "Стоимость по меню",
        "Фактическая цена реализации за ед", "Продано количество",
        "Подарено по акции", "Количество продаж с уценкой",
        "Кол-во товаров проданных по спец цене ПЛ", "Кол-во товаров проданных за живчики",
        "Кол-во списаний", "CustomersInMonh"
    ]

    df.drop(clean, axis=1, inplace=True)

    return df

