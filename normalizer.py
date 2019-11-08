import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.spatial import distance

# -------------Encoding and scaling------------
def createGroups5NY(row):
    if 0 < row['price'] <= 70:
        val = 0
    elif 70 < row['price'] <= 100:
        val = 1
    elif 100 < row['price'] <= 130:
        val = 2
    elif 130 < row['price'] <= 180:
        val = 3
    else:
        val = 4
    return val
def createGroups12NY(row):
    if 0 < row['price'] <= 50:
        val = 0
    elif 50 < row['price'] <= 65:
        val = 1
    elif 65 < row['price'] <= 75:
        val = 2
    elif 75 < row['price'] <= 85:
        val = 3
    elif 85 < row['price'] <= 99:
        val = 4
    elif 99 < row['price'] <= 115:
        val = 5
    elif 115 < row['price'] <= 125:
        val = 6
    elif 125 < row['price'] <= 135:
        val = 7
    elif 135 < row['price'] <= 149:
        val = 8
    elif 149 < row['price'] <= 165:
        val = 9
    elif 165 < row['price'] <= 190:
        val = 10
    elif 190 < row['price'] <= 245:
        val = 11
    else:
        val = 12
    return val
def createGroups5BCN(row):
    if 0 < row['price'] <= 35:
        val = 0
    elif 35 < row['price'] <= 60:
        val = 1
    elif 60 < row['price'] <= 100:
        val = 2
    elif 100 < row['price'] <= 140:
        val = 3
    else:
        val = 4
    return val
def createGroups3BCN(row):
    if 0 < row['price'] <= 60:
        val = 0
    elif 60 < row['price'] <= 120:
        val = 1
    else:
        val = 2
    return val


def normalize_csv(newDf):
    fields = ['neighbourhood_cleansed', 'neighbourhood', 'latitude', 'longitude', 'room_type',
              'host_response_time', 'host_is_superhost', 'number_of_reviews', 'reviews_per_month',
              'review_scores_rating',
              'property_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'guests_included',
              'availability_365', 'price']
    newDf = newDf[fields].copy()

    newDf['host_response_time'] = newDf['host_response_time'].astype(str)
    newDf['neighbourhood'] = newDf['neighbourhood'].astype(str)
    newDf['price'] = newDf['price'].map(lambda x: x.lstrip('$'))
    newDf['price'] = newDf['price'].str.replace(",", "")

    newDf['host_is_superhost'] = newDf['host_is_superhost'].replace(np.nan, '', regex=True)

    la = preprocessing.LabelEncoder()
    la.fit(newDf['host_response_time'])
    newDf['host_response_time'] = la.transform(newDf['host_response_time']).astype(float)
    la.fit(newDf['host_is_superhost'])
    newDf['host_is_superhost'] = la.transform(newDf['host_is_superhost']).astype(float)
    la.fit(newDf['neighbourhood_cleansed'])
    newDf['neighbourhood_cleansed'] = la.transform(newDf['neighbourhood_cleansed']).astype(float)
    la.fit(newDf['neighbourhood'])
    newDf['neighbourhood'] = la.transform(newDf['neighbourhood']).astype(float)
    la.fit(newDf['property_type'])
    newDf['property_type'] = la.transform(newDf['property_type']).astype(float)
    la.fit(newDf['room_type'])
    newDf['room_type'] = la.transform(newDf['room_type']).astype(float)
    la.fit(newDf['bed_type'])
    newDf['bed_type'] = la.transform(newDf['bed_type']).astype(float)

    newDf['latitude'] = newDf['latitude'].astype(float)
    newDf['longitude'] = newDf['longitude'].astype(float)
    newDf['number_of_reviews'] = newDf['number_of_reviews'].astype(float)
    newDf['reviews_per_month'] = newDf['reviews_per_month'].astype(float)
    newDf['accommodates'] = newDf['accommodates'].astype(float)
    newDf['bedrooms'] = newDf['bedrooms'].astype(float)
    newDf['beds'] = newDf['beds'].astype(float)
    newDf['guests_included'] = newDf['guests_included'].astype(float)
    newDf['availability_365'] = newDf['availability_365'].astype(float)
    newDf['number_of_reviews'] = newDf['number_of_reviews'].astype(float)
    newDf['review_scores_rating'] = newDf['review_scores_rating'].astype(float)
    newDf['price'] = newDf['price'].astype(float)

    # newDf = newDf[['neighbourhood_group','neighbourhood','latitude', 'longitude', 'room_type',
    #         'host_response_time', 'host_is_superhost', 'number_of_reviews', 'reviews_per_month', 'review_scores_rating',
    #         'accommodates', 'bathrooms', 'bedrooms', 'beds', 'beds_type', 'guests_included',
    #         'availability_365' , 'price']]

    # min_max_scaler = preprocessing.MinMaxScaler()
    # newDf.iloc[:, 0:12] = min_max_scaler.fit_transform(newDf.iloc[:, 0:12])
    newDf['price_cat'] = newDf.apply(createGroups3BCN, axis=1)

    newDf = newDf.fillna(newDf.mean())

    return newDf;
# -----------------------------------------------


