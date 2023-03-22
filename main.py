import requests
import json
from datetime import datetime
import csv
import numpy as np
import keys

# API key
client_id = keys.client_id
client_secret = keys.client_secret

body = {
    'client_id': client_id,
    'client_secret': client_secret,
    "grant_type": 'client_credentials'
}
r = requests.post('https://id.twitch.tv/oauth2/token', body)

keys = r.json()

headers = {
    "Client-ID": client_id,
    'Authorization': 'Bearer ' + keys['access_token']
}


game_name = "Overwatch 2"

url = "https://api.twitch.tv/helix/games"

params = {
    "name": game_name
}
response = requests.get(url, params=params, headers=headers)
data = json.loads(response.text)
game_id = (data['data'][0]['id'])

url = "https://api.twitch.tv/helix/streams"

params = {
    "game_id": game_id,
    "language": 'pl',
    "first": 100
}
                            
response = requests.get(url, params=params, headers=headers)

# JSON data parse
data = json.loads(response.text)

last_temp = ''
try:
    for i in range(50):
        print("Currently on site: ",i+1)
        last = data['pagination']['cursor']
        if last != last_temp:
            params = {
                "game_id": game_id,
                "language": 'pl',
                "first": 100,
                "after": last
                }
            response = requests.get(url, params=params, headers=headers)
            data['data'].extend(json.loads(response.text)['data'])
            last_temp = last
        else:
            break
except:
    pass

total = 0
viewers = []
channel_names =[]
tags =[]
titles = []
beggining = []
for i in range (len(data['data'])):
    total = total + data['data'][i]['viewer_count']
    viewers.append(data['data'][i]['viewer_count'])
    tags.append(data['data'][i]['tags'])
    channel_names.append(data['data'][i]['user_name'])
    titles.append(data['data'][i]['title'])
    beggining.append(data['data'][i]['started_at'])
print(f"Amount of streams for {game_name}: {len(data['data'])}")
print("Total viewers:", total)

outliers=[]

if len(viewers) == 0:
    bias_ratio = 0
    outliers = []
elif len(viewers) == 1:
    bias_ratio = 1
    outliers = viewers
elif len(viewers) == 2:
    if viewers[0]/(viewers[0] + viewers[1]) > 0.8:
        outliers.append(viewers[0])
        bias_ratio = sum(outliers)/total
    else:
        bias_ratio = 0
        outliers = []
elif len(viewers) == 3:
    average = np.mean(viewers)
    std_dev = np.std(viewers)
    z_scores = (viewers - average) / std_dev
    for i in range(len(z_scores)):
        if z_scores[i] > 1:
            outliers.append(viewers[i])
    bias_ratio = sum(outliers)/total

else:
    try:
        q1 = np.percentile(viewers, 25)
        q3 = np.percentile(viewers, 75)
        iqr = q3 - q1

        outliers = [x for x in viewers if x > q3 + 10*iqr]

        #print(outliers)
        bias_ratio = sum(outliers)/total 
    except:
        bias_ratio = 0
        outliers = []

print(viewers)
tags_done = []
channel_names_done = []
titles_done = []
beggining_done = []
try:
    for i in outliers:
        temp = viewers.index(i)
        result = [j for j in range(temp, len(viewers)) if viewers[j] == i]  # see how many times the same number occurs
        print(result)
        for j in result:
            if channel_names[j] not in channel_names_done:
                tags_done.append(tags[j])
                channel_names_done.append(channel_names[j])
                titles_done.append(titles[j])
                beggining_done.append(beggining[j])
except:
    pass

print(channel_names_done)

now = datetime.now()
formatted_now = now.strftime("%d-%m-%Y %H:%M:%S.%f")[:-7]

a = f"{formatted_now};{len(data['data'])};{total};{len(outliers)};{bias_ratio};{tags_done};{channel_names_done};{titles_done};{beggining_done}"

with open("OV2.csv", 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow(a.split(";"))
    file.flush()
    file.close()