import pandas as pd

from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:postgres@localhost:5432/optuna')

train = pd.read_csv("prefect-tutorial/src/database/train.csv")
test = pd.read_csv("prefect-tutorial/src/database/test.csv")

test['transaction_real_price'] = None

data = pd.concat([train,test], axis=0)

data.to_sql(name='apartments', con=engine, if_exists='fail', index=False, schema='public')
