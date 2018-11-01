from bs4 import BeautifulSoup
import urllib.request
import imghdr
import os

game_ids = ['252490', '440', '624090', '346110', '230410', '359550', '271590', '730', '570', '578080']

for game_id in game_ids:
    os.makedirs(os.path.join('screenshots', game_id), exist_ok=True)


    def download_screenshots(url):
        number_of_images = 0
        while number_of_images == 0:
            html = urllib.request.urlopen(url).read()
            soup = BeautifulSoup(html, features='html.parser')
            image_urls = soup.find_all('img', {'class': 'apphub_CardContentPreviewImage'})
            number_of_images = len(image_urls)

            for image_url in image_urls:
                image_type = imghdr.what('怕.jpg', urllib.request.urlopen(image_url['src']).read())
                print(image_type)
                if image_type == 'jpeg':
                    image_name = image_url['src'].split('/')[4] + '.' + image_type
                    print(image_name)
                    urllib.request.urlretrieve(image_url['src'], os.path.join('screenshots', game_id, image_name))
                    print('save image\n')


    browsefilters = ['toprated', 'trendday', 'trendweek', 'trendthreemonths', 'trendsixmonths', 'trendyear',
                     'mostrecent']

    for browsefilter in browsefilters:
        print(browsefilter)
        download_screenshots('https://steamcommunity.com/app/%s/screenshots/?browsefilter=%s' % (game_id, browsefilter))
