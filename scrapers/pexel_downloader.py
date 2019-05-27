import requests
import os
import shutil
import urllib.request

class PexelDownloader:
    def download_pexels_query(self, query, output_dir, max_imgs=2500):
        try:
            os.mkdir(output_dir)
        except:
            shutil.rmtree(os.path.abspath(output_dir))
            os.mkdir(output_dir)

        pexels_auth = {
            'Authorization': '563492ad6f91700001000001b79e9f89284a4b82a41214e486d81017'
        }

        r = requests.get(
            'https://api.pexels.com/v1/search?query=%s' % query,
            headers=pexels_auth
        ).json()

        c = 0

        while (c < max_imgs and r['next_page']):
            print('\n%d / %d' % (c, max_imgs))
            for photo in r['photos']:
                img = requests.get(
                    photo['src']['original'], stream=True)
                if photo['photographer']:
                    photographer = photo['photographer'].replace(' ', '-').lower() 
                else:
                    photographer = 'unknown'
                if img.status_code == 200:
                    with open('%s/%s_%d.jpg' % (output_dir, photographer, c), 'wb') as f:
                        img.raw.decode_content = True
                        shutil.copyfileobj(img.raw, f)
                    c += 1
                    print('.', end='', flush=True)
                else:
                    print('e', end='', flush=True)
            
            r = requests.get(r['next_page'], headers=pexels_auth).json()
