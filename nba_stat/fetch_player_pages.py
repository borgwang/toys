from selenium import webdriver
import pickle


# load pages database
try:
    player_pages = pickle.load(open('./data/player_pages.p', 'rb'))
except Exception as e:
    print('Erro loading exiting player_pages:', e)
    print('Starting from an empty player_pages')
    player_pages = {}
print('%d pages in database' % len(player_pages))

# chrome_options = webdriver.ChromeOptions()
# prefs = {"profile.managed_default_content_settings.images": 2}
# chrome_options.add_experimental_option("prefs", prefs)
# driver = webdriver.Chrome(chrome_options=chrome_options)
driver = webdriver.Chrome()
driver.set_page_load_timeout(5)

# load player page links
player_links = pickle.load(open('./data/player_links.p', 'rb'))
base_url = 'http://www.basketball-reference.com'

num_added, num_skipped = 0, 0

for name, link in player_links.items():
    if name not in player_pages:
        url = base_url + link
        try:
            driver.get(url)
        except Exception:
            pass
        finally:
            try:
                html_source = driver.page_source
                if html_source:
                    player_pages[name] = html_source
                    num_added += 1
                    print('load %s into menory' % name)
            except Exception as e:
                print('load error')
                raise e
    else:
        num_skipped += 1

    if num_added > 1 and num_added % 10 == 0:
        # dump to file player_pages.p
        with open('./data/player_pages.p', 'wb') as f:
            pickle.dump(player_pages, f, -1)

        print('Add %d pages. Skipped %d pages. Already had %d in database' % (
            num_added, num_skipped, len(player_pages)))
