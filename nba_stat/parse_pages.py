from bs4 import BeautifulSoup
import pickle


def read_table(table):
    if not table:
        return None

    stats_all_years = []
    for row in table.tbody.children:
        if row != '\n' and row.text != '':
            stats_per_year = []
            for col in row.children:
                if col != '\n' and col.text != '':
                    stats_per_year.append(str(col.text.encode('utf-8')))
            stats_all_years.append(stats_per_year)
    return stats_all_years


basic_avg_stats = {}
basic_total_stats = {}
per_minute_stats = {}
per_poss_stats = {}
advanced_stats = {}
player_pages = pickle.load(open('./data/player_pages.p', 'rb'))
for name, page in player_pages.items():
    soup = BeautifulSoup(page, 'lxml')

    # player basic stats per game
    table = soup.find(id='per_game')
    basic_avg_stats[name] = read_table(table)
    # player basic stats in total
    table = soup.find(id='totals')
    basic_total_stats[name] = read_table(table)
    # per_minute
    table = soup.find(id='per_minute')
    per_minute_stats[name] = read_table(table)
    # per_poss
    table = soup.find(id='per_poss')
    per_poss_stats[name] = read_table(table)
    # per_poss
    table = soup.find(id='per_poss')
    per_poss_stats[name] = read_table(table)
    # advanced
    table = soup.find(id='advanced')
    advanced_stats[name] = read_table(table)

    print('%s loaded.' % name)
import pdb; pdb.set_trace()
