import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from monte_carlo import mc_main
from data import coinmarketcap
from monte_carlo.utils import messaging, logging
import package_manager
import traceback

scheduler = BlockingScheduler(timezone=pytz.utc)


@scheduler.scheduled_job('cron', hour=0, minute=15, id='day', misfire_grace_time=600)
def daily_routine():
    try:
        package_manager.reload_packages()
        coinmarketcap.update()
        mc_main.multi_model_predict()
        mc_main.multi_model_retrain()
        logging.stop_logging()
    except Exception as e:
        messaging.notify_exception()
        print(traceback.format_exc())
        logging.stop_logging()
        raise e


daily_routine()
scheduler.start()
