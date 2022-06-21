# import starter
import requests

date_to_predict = {
    "year": 2021,
    "month": 3
}
print(date_to_predict)
url = 'http://localhost:9696/predict'

# url = 'http://172.31.7.246/starter'
# url ='http://127.0.0.1/starter:9696'
response = requests.post(url, json=date_to_predict)
# print( response)
print(response.json())
# preds = starter.run(date_to_predict)
# print(preds)