import os
import tarfile
from six.moves import urllib

FOOD_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
FOOD_PATH = "./dataset"


def fetch_data(url, path):
    tgz_path = os.path.join(path, "food101.tgz")
    urllib.request.urlretrieve(url, tgz_path)
    tgz = tarfile.open(tgz_path)
    tgz.extractall(path=path)
    tgz.close()


if __name__ == '__main__':
    fetch_data(FOOD_URL, FOOD_PATH)
