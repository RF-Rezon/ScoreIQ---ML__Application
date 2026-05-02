import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"   # Example ---->  LOG_FILE = "05_01_2026_13_30_00.log"

logs_dir = os.path.join(os.getcwd(), "logs")    # Example ---->  logs_dir = "C:/project/logs"
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)    # Example ---->  LOG_FILE_PATH = "C:/project/logs/05_01_2026_13_30_00.log"

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)