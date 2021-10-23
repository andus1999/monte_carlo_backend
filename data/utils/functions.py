import datetime


def get_current_timestamp():
    current_datetime = datetime.datetime.combine(datetime.date.today(),
                                                 datetime.time(0, 0, 0, 0, datetime.timezone.utc))
    current_timestamp = datetime.datetime.timestamp(current_datetime)
    return current_timestamp


def get_yesterday_datetime():
    dt = datetime.datetime.combine(datetime.date.today() - datetime.timedelta(days=1),
                                   datetime.time(0, 0, 0, 0, datetime.timezone.utc))
    return dt
