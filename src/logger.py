import logging
from datetime import datetime
import os

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs")
print(LOG_PATH)
os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILEPATH = os.path.join(LOG_PATH, LOG_FILE)

logging.basicConfig(
    level=logging.INFO, 
    filename=LOG_FILEPATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)