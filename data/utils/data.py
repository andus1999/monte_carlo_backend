

def get_historical_data_json(data):
    json_data = []
    for date in data:
        json_data.append({
            'open': date[0],
            'high': date[1],
            'low': date[2],
            'close': date[3],
            'volume': date[4],
            'market_cap': date[5],
            'timestamp': date[6],
        })
    return json_data
