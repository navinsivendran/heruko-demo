import requests

url = 'http://127.0.0.1:5000/results'
r = requests.post(url,json={'Ca':1,'P':2,'Ph':3,'SOC':4,'sand':5,'temperature':35,'humidity':25,'mositure soil':28,'soil temp':30})

print(r.json())