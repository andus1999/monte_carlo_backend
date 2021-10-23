"""Firebase messaging module."""

from firebase_admin import messaging
import json
import firebase_admin
from firebase_admin import credentials
import os
from ..resources import strings, filepaths


def send_message(title, body):
    firebase_tokens_path = os.path.join(os.path.dirname(__file__), filepaths.firebase_tokens)
    with open(firebase_tokens_path) as f:
        firebase_tokens = json.load(f)['firebase_tokens']

    try:
        firebase_admin.get_app()
    except ValueError:
        fb_key_path = os.path.join(os.path.dirname(__file__), filepaths.firebase_credentials)
        cred = credentials.Certificate(fb_key_path)
        firebase_admin.initialize_app(cred)

    message = messaging.MulticastMessage(
        notification=messaging.Notification(
            title=title,
            body=body,
        ), tokens=firebase_tokens)

    response = messaging.send_multicast(message)
    # Response is a message ID string.
    print('Successfully sent message:', response)


def notify_predictions_updated():
    send_message(strings.notification_prediction_title, strings.notification_prediction_body)


def notify_training_done():
    send_message(strings.notification_train_title, strings.notification_train_body)


def notify_exception():
    send_message(strings.notification_exception_title, strings.notification_exception_body)
