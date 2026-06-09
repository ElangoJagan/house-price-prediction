import logging
import os
from datetime import datetime

# create a unique logfile name using current timestamp 
LOG_FILE = f'{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log'

#step2: build paths to logs/folder and create it 
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)


# Step 3: full path to the log file 
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Step 4: Configure the Logging System:
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


##Step 5: create logger object  - this gets imported everywhere
logger = logging.getLogger('HousePriceLogger')