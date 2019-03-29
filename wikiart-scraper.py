import random
import time
import re
import os
import argparse
import urllib
import urllib.request
import itertools
import bs4
from bs4 import BeautifulSoup
import multiprocessing
from multiprocessing.dummy import Pool

class WikiartScraper:
    def get_painting_urls(self, wikiart_profile_url):
        try:
            # random sleep to decrease concurrence of requests
            url = wikiart_profile_url
            # "https://www.wikiart.org/en/paintings-by-%s/%s/%d" % (typep, searchword, count)
            soup = BeautifulSoup(urllib.request.urlopen(url), "lxml")
            regex = r'https?://uploads[0-9]+[^/\s]+/\S+\.jpg'
            url_list = re.findall(regex, str(soup.html()))
            return url_list
        except Exception as e:
            print('failed to scrape %s' % url, e)

    def downloader(self, links, output_dir):
        savepath = output_dir
        for link in links:
            try:
                time.sleep(0.2)  # try not to get a 403
                regex = r'((.(?<!\/))+$)'
                print(re.findall(regex, link))[0][0]
                urllib.request.urlretrieve(link, savepath + re.findall(regex, link)[0][0])
            except Exception as e:
                print("failed downloading ", e)
        
if __name__ == '__main__':
    ws = WikiartScraper()
    urls = ws.get_painting_urls("https://www.wikiart.org/en/profile/5c9ba655edc2c9b87424edfe/albums/favourites")
    ws.downloader(urls, './select_train/')
