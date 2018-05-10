from urllib.request import urlretrieve

with open("summit_post_urls_selected.txt", 'r') as urlfile:
    for index, url in enumerate(urlfile):
        filename = "./images/img_" + str(index) + ".jpg"
        urlretrieve(url.strip(), filename)
