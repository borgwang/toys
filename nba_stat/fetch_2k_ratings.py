from bs4 import BeautifulSoup
import urllib
import numpy as np
import pickle


# ---------- NBA 2K players rating ----------
# script: fetch_2k_ratings.py
# file saved to './data/2k_rating_14to18.p'
# format: dictionary. {name: [18, 17, 16, 15, 14]}
# note: missing value was padded with 0.

url_18 = 'https://www.sbnation.com' \
         '/nba/2017/8/12/16138230/nba-2k18-player-ratings-list'
page = urllib.urlopen(url_18)
html = page.read()
soup = BeautifulSoup(html, 'lxml')
table = soup.find_all('table')[0]
stats_18 = []
for tr in table.tbody.children:
    for td in tr:
        if td != '\n' and td.get_text() != '':
            item = td.get_text().encode('utf-8')
            stats_18.append(item)
stats_18 = np.reshape(stats_18, (-1, 2))

data_18 = {}
for i in stats_18:
    name = i[0].lower().replace('-', ' ')
    if name == 'steph curry':
        name = 'stephen curry'
    rating = int(i[1])
    data_18[name] = rating

data_17161514 = {}
for line in open('2k_14151617.txt'):
    l = line[:-1].split('\t')
    name = l[0].lower().replace('-', ' ')
    rating_list = [0]
    l += ['0']*(5-len(l))  # zero padding
    rating_list += [int(r) for r in l[1:]]
    data_17161514[name] = rating_list

# combine 18 with 17161514's ratings
for k, v in data_18.items():
    if k in data_17161514.keys():
        data_17161514[k][0] = v

# dump to  file
with open('./data/player_ratings_14to18.p', 'wb') as f:
    pickle.dump(data_17161514, f, -1)
