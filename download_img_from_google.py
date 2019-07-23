import urllib.request as urlreq
import os
# for catching errors
import traceback
import logging

# folder where you want to store downloaded images (change to your folder path)
folder_path = r'C:\Users\jadwi\training_fish'
# first path for goldfish img urls, second path for perch img urls
gf_path = r"urlsGoldfish.txt"
perch_path = r'urlsPerch.txt'


def download_img(url_path):
    rows = open(url_path).read().strip().split('\n')
    return rows


def get_images(name_of_fish, file_with_urls, folder_for_training_set):
    counter = 1
    excCounter = 0

    for url in file_with_urls:
        try:
            unique_name = ".".join((name_of_fish + str(counter), "jpg"))
            counter += 1
            path_name = os.path.join(folder_for_training_set, unique_name)
            print(path_name)
            # insert img downloaded from url into folder "training_fish"
            urlreq.urlretrieve(url, path_name)
        except Exception as e:
            excCounter += 1
            logging.error(traceback.format_exc())
    return excCounter


urls_gf = download_img(gf_path)
print(len(urls_gf))
# urls_perch = download_img(perch_path)
# get_images("gf", urls_gf, folder_path)
# get_images("perch", urls_perch, folder_patch)

