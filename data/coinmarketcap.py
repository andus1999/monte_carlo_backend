import csv
import datetime
import json
import os
import time

import cbpro
import selenium
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from monte_carlo.utils import logging
from monte_carlo.resources import strings
from . import binance
from .resources import filepaths
from .data_management import firestore, storage


def get_coinbase_ticker_list():
    client = cbpro.PublicClient()
    products = client.get_products()
    coinbase_ticker_list = []
    for product in products:
        ticker = product['base_currency']
        if ticker not in coinbase_ticker_list:
            coinbase_ticker_list.append(ticker)
    return coinbase_ticker_list


def month_to_number(string):
    string = string.lower()
    m = 0
    if string == 'jan':
        m = 1
    elif string == 'feb':
        m = 2
    elif string == 'mar':
        m = 3
    elif string == 'apr':
        m = 4
    elif string == 'may':
        m = 5
    elif string == 'jun':
        m = 6
    elif string == 'jul':
        m = 7
    elif string == 'aug':
        m = 8
    elif string == 'sep':
        m = 9
    elif string == 'oct':
        m = 10
    elif string == 'nov':
        m = 11
    elif string == 'dec':
        m = 12
    else:
        raise ValueError
    return m


def wait():
    time.sleep(loading_time)


def get_variable_name(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def get_api_historical_data(data):
    json_data = []
    for entry in data:
        date = datetime.datetime(int(entry[8]), int(entry[7]), int(entry[6]), 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
        json_data.append([float(price) for price in entry[0:6]]+[date.timestamp()])
    json_data.reverse()
    return json_data


def save_csv(name, data):
    json_data = get_api_historical_data(data)
    firestore.upload_historical_data(name, json_data[-30:])
    with open(os.path.join(os.path.dirname(__file__), filepaths.coins_json_path) + name + '.json', 'w+') as file:
        json.dump(json_data, file)
    storage.upload_historical_data(name)
    with open(os.path.join(os.path.dirname(__file__), filepaths.coins_csv_path) + name + '.csv', 'w') as file:
        write = csv.writer(file, quoting=csv.QUOTE_NONE)
        write.writerows(data)


def open_csv(name):
    with open(os.path.join(os.path.dirname(__file__), filepaths.coins_csv_path) + name + '.csv', 'r') as file:
        read = filter(None, csv.reader(file))
        data = list(read)
    return data


# noinspection PyGlobalUndefined
def initialize_x_paths():
    global historical_data
    historical_data = '/html/body/div/div[1]/div[1]/div[2]/div/div[2]/div/span/a[3]'
    global date_range
    date_range = '/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/span/button'
    global month_selection_back
    month_selection_back = '/html/body/div/div/div[2]/div/div[3]/div[2]/div/div[1]/div/div/div[1]/div/div/div[1]/div[1]/div/button[1]'
    global load_more
    load_more = '/html/body/div/div[1]/div/div[2]/div/div[3]/div[2]/div/p[1]/button'
    global select_day
    select_day = '/html/body/div/div/div[2]/div/div[3]/div[2]/div/div[1]/div/div/div[1]/div/div/div[1]/div[1]/div/div[2]/div[2]/div[1]/div[7]'
    global month_selection_done
    month_selection_done = '/html/body/div/div/div[2]/div/div[3]/div[2]/div/div[1]/div/div/div[1]/div/div/div[2]/button[2]'
    global table_body
    table_body = '/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[2]/table/tbody'
    global table_body_reload
    table_body_reload = '/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[2]/table/tbody'


def get_date(index, table):
    last_date = ''
    last_day = str(table[index][6])
    last_month = str(table[index][7])
    last_year = str(table[index][8])
    if len(last_day) == 1:
        last_day = '0' + last_day
    if len(last_month) == 1:
        last_month = '0' + last_month
    last_date += last_day
    last_date += last_month
    last_date += last_year
    return last_date


def get_link(index):
    try:
        name = crypto_list[index][3]
    except IndexError:
        name = crypto_list[index][1].lower().replace(' ', '-').replace('.', '-')
    return name


def get_ticker(index):
    if crypto_list is None:
        raise ValueError('Client not initialized.')
    ticker = crypto_list[index][2]
    return ticker


def update_table(index, new_table):
    name = get_link(index)
    table = open_csv(name)
    last_date = get_date(0, table)
    end_index = 0
    row = 0
    for row in range(0, len(new_table)):
        if get_date(row, new_table) == last_date:
            end_index = row - 1
    if end_index >= 0:
        for row in range(0, end_index + 1):
            table.insert(row, new_table[row])
        if index % 100 == 0:
            print('\n'+crypto_list[index][1] + ' ' + str(row + 1) + ' rows updated. Index: ' + str(index))
    elif index % 100 == 0:
        print('\n'+crypto_list[index][1] + ' already up to date. Index: ' + str(index))
    save_csv(name, table)


def set_date_range(months, driver):
    driver.execute_script("window.scrollBy(0,10000);")
    time.sleep(1)
    for k in range(0, months):  # 120
        driver.find_element_by_xpath(load_more).click()
        time.sleep(1)
        driver.execute_script("window.scrollBy(0,10000);")
        time.sleep(0.5)
    wait()


def get_element(x_path, driver, index):
    time_out = 20
    try:
        element = WebDriverWait(driver, time_out).until(EC.presence_of_element_located((By.XPATH, x_path)))
    except TimeoutException:
        print('\nElement not found. Returning None. ' + get_variable_name(x_path, globals())[0] + ' ' + get_link(index))
        element = None

    return element


def selenium_click_on(x_path, driver, index):
    element = get_element(x_path, driver, index)
    time.sleep(1)
    if element is not None:
        element.click()
        return True
    else:
        return False


def get_table_data(index, driver, update_only: bool):
    rows = 0
    while_count = 0
    table_elements, elements_per_row = None, None
    while rows == 0:
        time.sleep(2)
        driver.execute_script("window.scrollBy(0,400);")
        time.sleep(1)
        try:
            table_element = get_element(table_body, driver, index)
            table_text = table_element.get_attribute('innerHTML')
        except (selenium.common.exceptions.StaleElementReferenceException, AttributeError):
            table_element = None
            table_text = None
        if table_element is None or table_text is None:
            print('Error while reading table data. Retrying.')
            return None

        soup = BeautifulSoup(table_text, 'html.parser')
        td_list = soup.find_all('td')
        table_elements = []
        for e in td_list:
            table_elements.append(e.get_text())
        elements_per_row = 7
        rows = len(table_elements) // elements_per_row
        while_count += 1
        if rows == 0:
            driver.refresh()
            time.sleep(5)
        if while_count > 100:
            raise TimeoutException

    if len(table_elements) % rows != 0:
        raise IndexError
    table = []
    for r in range(0, rows):
        row = []
        date = table_elements[r * elements_per_row]
        for c in range(1, elements_per_row):
            text = table_elements[r * elements_per_row + c]
            row.append(float(text.replace('$', '').replace(',', '').replace('<', '')))
        day = int(date[4:6])
        row.append(day)
        month = month_to_number(date[0:3])
        row.append(month)
        year = int(date[8:12])
        row.append(year)
        row.append(index)
        table.append(row)
    table = binance.get_table_for_coinmarketcap(table, index, update_only)
    return table


def go_to_historical_data(name, driver, index):
    driver.get('https://coinmarketcap.com/currencies/' + name)
    if selenium_click_on(historical_data, driver, index):
        return True
    else:
        return False


def get_historical_data(index, driver):
    start_time = time.time()
    name = get_link(index)

    go_to_historical_data(name, driver, index)

    set_date_range(history_in_years * 12, driver)

    table = get_table_data(index, driver, False)

    print(crypto_list[index][1] + ' ' + str(len(table)) + ' rows.')

    save_csv(name, table)

    print(crypto_list[index][1] + ' done. Index: ' + str(index))

    delta_time = round((time.time() - start_time))
    print('Delta Time: ' + str(delta_time) + ' s')


def update_historical_data(index, driver):
    start_time = time.time()
    name = get_link(index)
    if not os.path.exists(os.path.join(os.path.dirname(__file__), filepaths.coins_csv_path) + name + '.csv'):
        print('No file found: ' + name, end=' ')
        return
    new_table = None
    while new_table is None:
        if not go_to_historical_data(name, driver, index):
            print('No Historical Data Found: ' + name)
            return
        new_table = get_table_data(index, driver, True)
    update_table(index, new_table)

    delta_time = round((time.time() - start_time))
    if index % 100 == 0:
        print('Delta Time: ' + str(delta_time) + ' s')


def initialize(off_server=False):
    print('Starting...')
    initialize_x_paths()
    options = webdriver.ChromeOptions()
    if off_server:
        options.add_argument('--user-data-dir=C:\\Users\\Andi\\AppData\\Local\\Google\\Chrome\\User Data')
    else:
        options.add_argument('--user-data-dir=C:\\Users\\Andreas\\AppData\\Local\\Google\\Chrome\\User Data')

    options.add_argument('--profile-directory=Profile 2')

    with open(os.path.join(os.path.dirname(__file__), filepaths.crypto_names_csv), newline='') as f:
        reader = filter(None, csv.reader(f))
        global crypto_list
        crypto_list = list(reader)

    global loading_time
    loading_time = 10
    global history_in_years
    history_in_years = 10

    web_driver = webdriver.Chrome(options=options)
    return web_driver


def update(off_server=False):
    logging.switch_logging_category(strings.logging_coinmarketcap_update)
    web_driver = initialize(off_server)
    start_time = time.time()
    for i in range(0, len(crypto_list)):
        update_historical_data(i, web_driver)
    delta_time = time.time()-start_time
    print('\nCoinmarketcap update took {:.2f} hours.'.format(delta_time/3600))
    web_driver.close()


def update_coinbase_coins(off_server=False):
    logging.switch_logging_category(strings.logging_coinmarketcap_update)
    web_driver = initialize(off_server)
    start_time = time.time()
    coinbase_ticker_list = get_coinbase_ticker_list()
    for i in range(0, len(crypto_list)):
        if get_ticker(i) in coinbase_ticker_list:
            update_historical_data(i, web_driver)
    delta_time = time.time()-start_time
    print('\nCoinmarketcap update took {:.2f} hours.'.format(delta_time/3600))
    web_driver.close()


def add(indices, off_server=False):
    web_driver = initialize(off_server)
    for i in indices:
        get_historical_data(i, web_driver)
    print('Done.')
    web_driver.close()


def reload_historical_data(indices, off_server=False):
    web_driver = initialize(off_server)
    for i in indices:
        name = get_link(i)
        if os.path.exists(os.path.join(os.path.dirname(__file__), filepaths.coins_csv_path) + name + '.csv'):
            get_historical_data(i, web_driver)
        else:
            print('No file found {}.'.format(name))
    print('Done.')
    web_driver.close()
