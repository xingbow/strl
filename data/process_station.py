import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
from pyproj import Proj
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def k_cluster(K, X):
    model = KMeans(n_clusters=K).fit(X)
    label_ = model.labels_
    centers_ = model.cluster_centers_
    return label_, centers_


def geo_mapping(location):
    if not hasattr(geo_mapping, "Proj"):
        geo_mapping.Proj = Proj(init='epsg:32118')
    return geo_mapping.Proj(*location)


def get_episode(df, episode):
    df['starttime'] = pd.to_datetime(df['starttime'])
    return df[(episode[0] <= df['starttime'].dt.hour)
              & (df['starttime'].dt.hour <= episode[1])]


def preprocess(df):
    tdf = df[['tripduration', 'starttime',
              'start station id', 'end station id']].drop_duplicates()

    tdf.columns = ['duration', 'timpstamp', 'start', 'end']

    sgdf = df[['start station id',
               'start station name',
               'start station latitude',
               'start station longitude']].drop_duplicates()

    sgdf.columns = ['id', 'name', 'lat', 'long']

    egdf = df[['end station id',
               'end station name',
               'end station latitude',
               'end station longitude']].drop_duplicates()

    egdf.columns = ['id', 'name', 'lat', 'long']

    gdf = pd.concat([sgdf, egdf]).drop_duplicates()

    gdf['location'] = list(zip(gdf['long'],
                               gdf['lat']))

    gdf['location'] = gdf['location'].map(geo_mapping)
    gdf['x'] = gdf['location'].map(lambda x: x[0])
    gdf['y'] = gdf['location'].map(lambda x: x[1])

    del gdf['location'], gdf['long'], gdf['lat']
    gdf = gdf.reset_index(drop=True)

    return gdf, tdf


def main():
    df = pd.read_csv('./2014-02 - Citi Bike trip data.csv')
    gdf, tdf = preprocess(df)

    ldf = pd.read_csv('./stationStatus.csv')
    ldf.columns = ['id', 'name', 'limit']
    del ldf['name']

    df = pd.merge(gdf, ldf,
                  how='left', on='id')

    df = df.sort_values('id')
    df.to_csv('station.csv', index=False)


if __name__ == "__main__":
    main()
