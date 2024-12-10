import sys
import requests
url = 'https://ihr.iijlab.net/ihr/api/network_delay/'
params={'timebin__gte': '2017-06-01T00:00','timebin__lte': '2017-07-01T00:00'}
resp = requests.get(url,params)
if (resp.ok):
    data = resp.json()
    print(data)
else:
    resp.raise_for_status()