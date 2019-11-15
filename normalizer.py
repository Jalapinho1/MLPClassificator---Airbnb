import pandas as pd
import numpy as np
from sklearn import preprocessing

# -------------Encoding and scaling------------
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

    newDf['price_cat'] = newDf.apply(createGroups3BCN, axis=1)

    onehot = pd.get_dummies(newDf['price_cat'])
    onehot.columns = ['cheap', 'medium', 'expensive']
    newDf = newDf.drop('price_cat', axis=1)
    newDf = newDf.drop('price', axis=1)
    newDf = newDf.join(onehot)

    newDf = newDf.fillna(newDf.mean())

    return newDf;
# -----------------------------------------------
def normalize_csv_regression(newDf):
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

    newDf = newDf.drop(newDf[newDf.price > 500].index)

    newDf = newDf.fillna(newDf.mean())

    return newDf;
# -----------------------------------------------

