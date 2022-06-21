
import pickle
import pandas as pd
# import argparse
from flask import Flask, request, jsonify

def read_data(filename):

    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(features):

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    X = dv.transform(features)
    preds = lr.predict(X)   

    return preds

def build_result_df(df, year, month, preds):

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame({"Ride_id": df['ride_id'].tolist(), "Predicted":preds})

    return df_result

def save_df_parquet(df_result, year, month):

    df_result.to_parquet(
        f"results_taxi_{year}_{month}.parquet",
        engine='pyarrow',
        compression=None,
        index=False
    )

def run(args):

    # if type(args)==dict:
    year = int(args["year"])
    month = int(args["month"])
    # elif type(args)==argparse.Namespace:
    #     year = int(args.year)
    #     month = int(args.month)    

    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month:02d}.parquet')
    
    categorical = ['PUlocationID', 'DOlocationID']
    features = df[categorical].to_dict(orient='records')

    preds = predict(features)

    df_result = build_result_df(df, year, month, preds)

    save_df_parquet(df_result, year, month)
   
    print(preds.mean())
    return (preds.mean())

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    date_to_predict = request.get_json()
    print(date_to_predict)
    pred_mean = run(date_to_predict)
    print(pred_mean)
    result = {
        "Mean duration": pred_mean
    }
    
    return jsonify(result)


if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=9696)

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--year",
    #     help="year of data to be processed"
    # )
    # parser.add_argument(
    #     "--month",
    #     help="month of data to be processeed"
    # )
    # args = parser.parse_args()
    # print(args)
    # run(args)


