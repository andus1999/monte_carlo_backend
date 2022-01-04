import csv
import datetime
import json
import os
import time
from transformers import pipeline
from selenium import webdriver
from ..resources import filepaths
from . import x_paths
from monte_carlo.utils import logging
from monte_carlo.resources import strings


def save_json(data, coin_id):
    sentiment_list = load_sentiment(coin_id)
    sentiment_list.append(data)
    with open(os.path.join(os.path.dirname(__file__), filepaths.sentiment_path) + coin_id + '.json', 'w+') as file:
        json.dump(sentiment_list, file)


def load_sentiment(coin_id):
    path = os.path.join(os.path.dirname(__file__), filepaths.sentiment_path) + coin_id + '.json'
    if not os.path.exists(path):
        return []
    with open(os.path.join(os.path.dirname(__file__), filepaths.sentiment_path) + coin_id + '.json') as file:
        sentiment = json.load(file)
    return sentiment


def get_link(index):
    try:
        name = crypto_list[index][3]
    except IndexError:
        name = crypto_list[index][1].lower().replace(' ', '-').replace('.', '-')
    return name


def get_name(index):
    if crypto_list is None:
        raise ValueError('Client not initialized.')
    name = crypto_list[index][1]
    return name


def convert_score_to_string(score):
    if score > 0.1:
        return 'positive'
    if score < -0.5:
        return 'negative'
    return 'neutral'


def get_headline_elements(driver, index):
    driver.get(f'https://duckduckgo.com/?kax=v247-3&kad=en_US&q={get_name(index)}%20crypto&iar=news&ia=news')
    time.sleep(4)
    headline_elements = driver.find_elements_by_xpath(x_paths.headlines)
    return headline_elements


def analyse_headline(classifier, headline):
    sentence = headline+'.'
    result = classifier(sentence.lower())[0]
    sentiment_score = result['score']
    sentiment_value = result['label']
    sentiment = sentiment_score
    if sentiment_value == 'NEGATIVE':
        sentiment *= -1
    if sentiment_value == 'NEUTRAL':
        sentiment = 0
    return sentiment


def get_sentiment_data(classifier, headline_elements, index):
    headlines = []
    sentiments = []
    for element in headline_elements:
        headline = element.get_attribute('innerHTML')
        link = element.get_attribute('href')
        if not get_name(index).lower() in headline.lower():
            continue
        sentiment = analyse_headline(classifier, headline)
        sentiments.append(sentiment)
        headlines.append({
            'headline': headline,
            'sentiment_score': sentiment,
            'sentiment_value': convert_score_to_string(sentiment),
            'link': link,
        })

    if len(sentiments) == 0:
        average_sentiment = None
        sentiment_value = None
    else:
        average_sentiment = sum(sentiments)/len(sentiments)
        sentiment_value = convert_score_to_string(average_sentiment)
    print(f'{average_sentiment} {sentiment_value} {get_name(index)}')
    sentiment_info = {
        'sentiment_score': average_sentiment,
        'sentiment_value': sentiment_value,
    }
    data = {
        'sentiment': sentiment_info,
        'headlines': headlines,
        'timestamp': datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
    }
    return data


def update_news_sentiment():
    logging.switch_logging_category(strings.logging_sentiment)
    driver = webdriver.Chrome()
    classifier = pipeline('sentiment-analysis')
    for i in range(0, len(crypto_list)):
        headline_elements = get_headline_elements(driver, i)
        data = get_sentiment_data(classifier, headline_elements, i)
        save_json(data, get_link(i))
    driver.close()


with open(os.path.join(os.path.dirname(__file__), '../'+filepaths.crypto_names_csv), newline='') as f:
    reader = filter(None, csv.reader(f))
    crypto_list = list(reader)
