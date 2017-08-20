from bs4 import BeautifulSoup
import urllib
import pickle


# ---------- All players basic info ----------
# script: fetch_palyers_info.py
# file saved to './data/player_info.p'
# format: list.
# ['Status', 'Player', 'First year', 'Last year', 'Position',
# 'Height', 'Weight', 'Birth Date', 'College']


base_url = 'https://www.basketball-reference.com/players/'
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z']

player_info = []
num_players = 0
for letter in letters:
    url = base_url + letter + '/'
    page = urllib.urlopen(url)
    html = page.read()
    soup = BeautifulSoup(html, 'lxml')
    table = soup.find(id='players')

    mapping = {'year_min': 2, 'year_max': 3, 'pos': 4, 'height': 5,
               'weight': 6, 'birth_date': 7, 'college_name': 8}
    count = 0
    for item in table.tbody.children:
        if item != '\n' and item.get_text() != '':
            player = [None] * 9
            for col in item.children:
                if col.name == 'th':
                    name = str(col.text)
                    if col.strong is not None:
                        status = 'active'
                    elif '*' in name:
                        status = 'hall of fame'
                        name = name[:-1]
                    else:
                        status = 'retired'
                    player[0] = status
                    player[1] = name
                if col.name == 'td':
                    player[mapping.get(col.get('data-stat'))] = str(col.text)

            player_info.append(player)
            count += 1
    print('%s for %d players' % (letter.upper(), count))
    num_players += count
print('%d players in total' % num_players)

# dump to file
with open('./data/player_info.p', 'wb') as f:
    pickle.dump(player_info, f, -1)
