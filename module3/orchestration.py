from calendar import month
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import prefect
from prefect import task,flow
from prefect.task_runners import SequentialTaskRunner
from datetime import date , timedelta , datetime

import pickle

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = prefect.get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    logger = prefect.get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):

    logger =  prefect.get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_path(Date = None):

    """returns the path of training and 
    val data as per the date"""

    logger = prefect.get_run_logger()
    if Date == None:
        today = date.today()
        two_months_back = str(today - timedelta(days=62))[:7]
        one_month_back = str(today - timedelta(days=31))[:7]
        train_path = "data/" + "fhv_tripdata_" + two_months_back + ".parquet"
        val_path = "data/" + "fhv_tripdata_" + one_month_back + ".parquet"
        logger.info(f"train path: {train_path}")
        logger.info(f"val path: {val_path}")
        return train_path , val_path
    
    else:
        Date = datetime.strptime(Date,'%Y-%m-%d') 
        two_months_back = str(Date - timedelta(days=62))[:7]
        one_month_back = str(Date - timedelta(days=31))[:7]
        train_path = "data/" + "fhv_tripdata_" + two_months_back + ".parquet"
        val_path = "data/" + "fhv_tripdata_" + one_month_back + ".parquet"
        logger.info(f"train path: {train_path}")
        logger.info(f"val path: {val_path}")
        return train_path , val_path

    


@flow(task_runner=SequentialTaskRunner())
def main(Date :str = "2021-03-15" ):

    train_path ,val_path = get_path(Date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path).result()
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path).result()
    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    #Saving model and dict vectorizer
    with open(f"models/model-{Date}.pkl" , "wb") as f_out:
        pickle.dump(lr,f_out)

    with open(f"models/dv-{Date}.pkl" , "wb") as f_out:
        pickle.dump(dv,f_out)

if __name__ == "__main__":
    
    main(Date="2021-08-15")