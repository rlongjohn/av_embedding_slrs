import logging
from datetime import datetime

# set up logging for status updates
logging.basicConfig(level=logging.INFO,
                    filename="experiment" + datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + ".log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()