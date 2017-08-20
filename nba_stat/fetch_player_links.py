from bs4 import BeautifulSoup
import urllib
import pickle


# ---------- All players page links ----------
# script: fetch_palyer_links.py
# file saved to './data/palyer_links.p'
# format: dict.
# {name: link}

base_url = 'https://www.basketball-reference.com/players/'
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z']

player_links = {}
for letter in letters:
    url = base_url + letter + '/'
    page = urllib.urlopen(url)
    html = page.read()
    soup = BeautifulSoup(html, 'lxml')
    table = soup.find(id='players')
    count = 0
    for th in table.tbody.find_all(name='th'):
        name = str(th.text)
        if '*' in name:
            name = name[:-1]
        player_links[name] = th.a.get('href')
        count += 1
    print('%d player links added.' % count)

# dump to file
with open('./data/player_links.p', 'wb') as f:
    pickle.dump(player_links, f, -1)
