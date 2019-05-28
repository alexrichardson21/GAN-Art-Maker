import time
import re
import os
import urllib
import urllib.request
import itertools
import bs4
import shutil
import glob
from bs4 import BeautifulSoup
from selenium import webdriver

class WikiartScraper:
    def scrape_art(self, wikiart_url, output_dir):

        # Try opening webpage with selenium in Windows or Mac
        for filepath in glob.iglob('./chromedrivers'):
            try:
                driver = webdriver.Chrome(filepath)
                break
        
        if not driver:
            raise Exception('No working Chrome Driver')

        driver.get(wikiart_url)

        # Clicks 'LOAD MORE' until all pictures are loaded 
        while(True):
            try:
                time.sleep(10)
                driver.find_element_by_class_name(
            "masonry-load-more-button").click()
            except:
                break
        
        # Finds all title blocks from fully expanded webpage
        soup = BeautifulSoup(driver.page_source, "lxml")
        driver.close()
        title_blocks = soup.find_all('div', attrs={'class': 'title-block'})

        # Finds all external painting urls from title blocks
        painting_pages = []
        for block in title_blocks:
            painting_pages += re.findall(r'\/en\/\S+\/\S+\w', str(block))
        
        try:
            os.mkdir(output_dir)
        except:
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)

        print('Downloading %d images from wikiart ...' % len(painting_pages))
        self.download_largest_images(painting_pages, output_dir)
    
    def download_largest_images(self, painting_pages, output_dir):
        # # Make new folder
        # os.mkdir(output_dir)

        for i, page in enumerate(painting_pages):
            print('Painting: %d / %d' % (i + 1, len(painting_pages)))
            url = "https://www.wikiart.org%s" % str(page)
            print('Finding best image from %s ...' % url)

            try:
                # Finds the ArtworkViewCtrl block
                soup = BeautifulSoup(urllib.request.urlopen(url), "lxml")
                block = soup.find('main', attrs={'ng-controller': 'ArtworkViewCtrl'})

                # Finds all image urls for painting
                image_urls = re.findall(
                    r'https?://uploads[0-9]+[^/\s]+/\S+\.jpg', str(block))
                
                if image_urls:
                    # Finds the largest sized image of painting
                    image_sizes = [self.get_size(image) for image in image_urls]
                    largest_image = image_urls[image_sizes.index(max(image_sizes))]

                    # Downloads largest image
                    
                    self.download_image_to_dir(largest_image, output_dir)
                else:
                    print("no images")
            except Exception as e:
                print("failed downloading", e)
    
    def get_size(self, url):
        # Gets size of image from url in bytes
        file = urllib.request.urlopen(url)
        return len(file.read())
    
    def download_image_to_dir(self, link, output_dir):
        try:
            filename = re.findall(r'((.(?<!\/))+$)', link)[0][0]
            print('Downloading %s ...' % filename)
            urllib.request.urlretrieve(link, output_dir + '/' + filename)
            print()

        except Exception as e:
            print("failed downloading ", e)
        
if __name__ == '__main__':
    profile_url = 'https://www.wikiart.org/en/profile/5c9ba655edc2c9b87424edfe/albums/favourites'
    output_dir = './select_train/'
    
    ws = WikiartScraper()
    ws.scrape_art(profile_url, output_dir)
