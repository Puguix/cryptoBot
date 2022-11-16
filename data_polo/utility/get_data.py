import pandas as pd

def get_historical_from_db(exchange, symbol, timeframe):
    symbol = symbol.replace('/','-')
    df = pd.read_csv("/home/puguix/Desktop/cryptoBot/data_polo/database/"+exchange+"/"+timeframe+"/"+symbol+".csv")
    df = df.set_index(df['date'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['date']
    return df

def get_historical_from_path(path):
    df = pd.read_csv(filepath_or_buffer=path)
    df = df.set_index(df['date'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['date']
    return df