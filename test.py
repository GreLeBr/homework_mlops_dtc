# import starter
import requests

date_to_predict = {
    "year": 2021,
    "month": 3
}
print(date_to_predict)
url = 'http://localhost:9696/predict'

response = requests.post(url, json=date_to_predict)

print(response.json())
