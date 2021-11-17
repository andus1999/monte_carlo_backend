import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from data.resources import filepaths
import os


def upload_historical_data(coin):
    try:
        app = firebase_admin.get_app('decryptor')
    except ValueError:
        cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), filepaths.firebase_db_key))
        app = firebase_admin.initialize_app(cred, {
            'storageBucket': 'decryptor-329419.appspot.com',
            'projectId': 'decryptor-329419'
        }, 'decryptor')

    bucket = storage.bucket(None, app)
    blob = bucket.blob('historical_data/' + coin + '.json')
    blob.upload_from_filename(os.path.join(os.path.dirname(__file__), filepaths.storage_coins_path + coin + '.json'))
