import os
from datetime import date
from io import StringIO

import pandas as pd
from google.cloud import storage
from tqdm import tqdm


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def download():
    storage_client = storage.Client.from_service_account_json(find('secretgc_ip.json', '/home'))

    files = os.listdir('./Simulations')

    blobs = storage_client.list_blobs('ip-free')
    for blob in tqdm(blobs):
        if blob.name not in files:
            blob.download_to_filename(os.path.join('Simulations', blob.name))


def upload():
    storage_client = storage.Client.from_service_account_json(find('secretgc_ip.json', '/home'))
    bucket = storage_client.bucket('ip-free')

    files = os.listdir('.')
    files = [f for f in files if 'csv' in f]

    for file in tqdm(files):
        data = pd.read_csv(os.path.join('.', file))
        name = file.split('.')[0]
        f = StringIO()
        data.to_csv(f, index_label=False)
        f.seek(0)
        blob = bucket.blob(name + '_' + str(date.today()) + '.csv')
        blob.upload_from_file(f, content_type='text/csv')
