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
from selenium import webdriver

class WikiartScraper:
    def get_painting_urls(self, wikiart_profile_url):
        # random sleep to decrease concurrence of requests

        # "https://www.wikiart.org/en/paintings-by-%s/%s/%d" % (typep, searchword, count)
        # soup = BeautifulSoup(urllib.request.urlopen(url), "lxml")
        driver = webdriver.Chrome("./chromedriver73")
        driver.get(wikiart_profile_url)
        while(True):
            try:
                driver.find_element_by_class_name(
            "masonry-load-more-button").click()
                time.sleep(10)
            except Exception as e:
                break
        
        # regex = r'https?://uploads[0-9]+[^/\s]+/\S+\.jpg'
        # url_list = re.findall(regex, str(driver.page_source))
        soup = BeautifulSoup(driver.page_source, "lxml")
        driver.close()
        title_blocks = soup.find_all('div', attrs={'class': 'title-block'})
        painting_pages = []
        for block in title_blocks:
            # print(block.text)
            painting_pages += re.findall(r'\/en\/\S+\/\S+\w', str(block))
        return painting_pages
    
    def largest_images(self, painting_pages):
        urls = []
        for page in painting_pages:
            url = "https://www.wikiart.org" + str(page)
            print(url)
            soup = BeautifulSoup(urllib.request.urlopen(url), "lxml")
            block = soup.find('main', attrs={'ng-controller': 'ArtworkViewCtrl'})
            image_urls = re.findall(
                r'https?://uploads[0-9]+[^/\s]+/\S+\.jpg', str(block))
            if image_urls:
                image_sizes = [self.getsizes(image) for image in image_urls]
                urls.append(image_urls[image_sizes.index(max(image_sizes))])
        return urls
    
    def getsizes(self, uri):
        try:
            file = urllib.request.urlopen(uri)
            return len(file.read())
        except:
            return 0
    
    def downloader(self, links, output_dir):
        savepath = output_dir
        for link in links:
            try:
                time.sleep(0.2)  # try not to get a 403
                regex = r'((.(?<!\/))+$)'
                print(re.findall(regex, link)[0][0])
                urllib.request.urlretrieve(link, savepath + re.findall(regex, link)[0][0])
            except Exception as e:
                print("failed downloading ", e)
        
if __name__ == '__main__':
    ws = WikiartScraper()
    urls = ws.get_painting_urls("https://www.wikiart.org/en/profile/5c9ba655edc2c9b87424edfe/albums/favourites")
    images = ws.largest_images(urls)
    ws.downloader(images, './select_train/')
