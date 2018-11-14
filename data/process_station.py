import numpy as np
import pandas as pd
import os


def get_episode(df, durations):
    def bool_expr(duration):
        return lambda x: (duration[0] <= x) & (x < duration[1])

    def and_bool_exprs(exprs):
        def chained(x):
            ret = exprs[0](x)
            for expr in exprs[1:]:
                ret = ret | expr(x)
            return ret
        return chained

    if not isinstance(durations[0], list):
        durations = [durations]

    expr = and_bool_exprs([bool_expr(duration) for duration in durations])
    df['starttime'] = pd.to_datetime(df['starttime'])

    df = df[expr(df['starttime'].dt.hour)]

    assert len(df['episode'].unique()) == 1

    return df


# def preprocess(df):
#     tdf = df[['tripduration', 'starttime',
#               'start station id', 'end station id']].drop_duplicates()

#     tdf.columns = ['duration', 'timpstamp', 'start', 'end']

#     sgdf = df[['start station id',
#                'start station name',
#                'start station latitude',
#                'start station longitude']].drop_duplicates()

#     sgdf.columns = ['id', 'name', 'lat', 'long']

#     egdf = df[['end station id',
#                'end station name',
#                'end station latitude',
#                'end station longitude']].drop_duplicates()

#     egdf.columns = ['id', 'name', 'lat', 'long']

#     gdf = pd.concat([sgdf, egdf]).drop_duplicates()

#     gdf['location'] = list(zip(gdf['long'],
#                                gdf['lat']))

#     gdf['location'] = gdf['location'].map(geo_mapping)
#     gdf['x'] = gdf['location'].map(lambda x: x[0])
#     gdf['y'] = gdf['location'].map(lambda x: x[1])

#     del gdf['location'], gdf['long'], gdf['lat']
#     gdf = gdf.reset_index(drop=True)

#     return gdf, tdf


def extract_geo_info(df):
    gdf = df[['start station id',
              'start id',
              'rx1',
              'ry1']].drop_duplicates()

    gdf.columns = ['sid', 'rid', 'x', 'y']

    ldf = pd.read_csv('./stationStatus.csv')
    ldf.columns = ['sid', 'name', 'limit']
    del ldf['name']

    gdf = pd.merge(gdf, ldf, how='left', on='sid')

    gdf = gdf.groupby('rid').mean().reset_index(level=0)

    gdf['limit'] = gdf['limit'].round()

    gdf = gdf.reset_index(drop=True)

    gdf = gdf.sort_values('rid')

    del gdf['sid']

    return gdf


'''
Define 5 episode
episode0 = [7,11]
episode1 = [[11,12],[16,17]]
episode2 = [12,16]
episode3 = [17,18]
episode4 = [18,23]'''


def main():
    org_df = pd.read_csv('./test.csv')

    episodes = [
        [7, 11],
        [[11, 12], [16, 17]],
        [12, 16],
        [17, 18],
        [18, 23],
    ]

    os.makedirs('geo', exist_ok=True)

    for ix, episode in enumerate(episodes):
        df = get_episode(org_df, episode)

        gdf = extract_geo_info(df)

        gdf.to_csv('geo/{}.csv'.format(ix), index=False)


if __name__ == "__main__":
    main()
