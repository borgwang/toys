from bs4 import BeautifulSoup
from bs4 import Comment
import urllib
import pickle
import threading
import time
import Queue


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


base_url = 'https://www.basketball-reference.com'

basic_avg_stats = {}
basic_total_stats = {}
per_minute_stats = {}
per_poss_stats = {}
advanced_stats = {}
player_links = pickle.load(open('./data/player_links.p', 'rb'))


class Parser(object):

    def __init__(self, queue, data):
        self.queue = queue
        self.data = data

    def run(self):
        while True:
            if not self.queue.empty():
                name, link = self.queue.get(block=True)
            else:
                return
            url = base_url + link
            page = urllib.urlopen(url)
            html = page.read()
            main_soup = BeautifulSoup(html, 'lxml')

            comment_str = ''
            all_comments = main_soup.find_all(
                string=lambda text: isinstance(text, Comment))
            for c in all_comments:
                comment_str += str(c.extract().encode('utf-8'))

            comment_soup = BeautifulSoup(comment_str, 'lxml')

            # player basic stats per game
            table = main_soup.find(id='per_game')
            self.data['basic_avg_stats'][name] = read_table(table)
            # player basic stats in total
            table = comment_soup.find(id='totals')
            self.data['basic_total_stats'][name] = read_table(table)
            # per_minute
            table = comment_soup.find(id='per_minute')
            self.data['per_minute_stats'][name] = read_table(table)
            # per_poss
            table = comment_soup.find(id='per_poss')
            self.data['per_poss_stats'][name] = read_table(table)
            # per_poss
            table = comment_soup.find(id='per_poss')
            self.data['per_poss_stats'][name] = read_table(table)
            # advanced
            table = comment_soup.find(id='advanced')
            self.data['advanced_stats'][name] = read_table(table)

            print('%s readed. ' % name)


threads = []
q = Queue.Queue()
table_data = {
    'basic_avg_stats': {}, 'basic_total_stats': {}, 'per_minute_stats': {},
    'per_poss_stats': {}, 'advanced_stats': {}}

for i in player_links.items():
    q.put(i)

for _ in range(1):
    parser = Parser(q, table_data)
    t = threading.Thread(target=parser.run)
    threads.append(t)

for t in threads:
    t.start()

for t in threads:
    t.join()


# ------------
# single thread
# ------------
# for name, link in player_links.items():
#     url = base_url + link
#     page = urllib.urlopen(url)
#     html = page.read()
#     main_soup = BeautifulSoup(html, 'lxml')
#
#     comment_str = ''
#     all_comments = main_soup.find_all(
#         string=lambda text: isinstance(text, Comment))
#     for c in all_comments:
#         comment_str += str(c.extract().encode('utf-8'))
#
#     comment_soup = BeautifulSoup(comment_str, 'lxml')
#
#     # player basic stats per game
#     table = main_soup.find(id='per_game')
#     basic_avg_stats[name] = read_table(table)
#     # player basic stats in total
#     table = comment_soup.find(id='totals')
#     basic_total_stats[name] = read_table(table)
#     # per_minute
#     table = comment_soup.find(id='per_minute')
#     per_minute_stats[name] = read_table(table)
#     # per_poss
#     table = comment_soup.find(id='per_poss')
#     per_poss_stats[name] = read_table(table)
#     # per_poss
#     table = comment_soup.find(id='per_poss')
#     per_poss_stats[name] = read_table(table)
#     # advanced
#     table = comment_soup.find(id='advanced')
#     advanced_stats[name] = read_table(table)
#
#     print('%s readed. ' % name)
