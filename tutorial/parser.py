import json
from logger import logger

# Open the config file
try:
    file = open("config.json")
except FileNotFoundError:

    logger.error("Config file not found.")
    logger.error("Make sure a \"config.json\" file exists. \n"
                "See the docs at:\n"
                "https://kostasvaleckas.github.io/PyLongslit/")
    
    exit()
    
logger.info("Config file found. Loading user parameters...")

data = json.load(file)

# Define parameter groups for easier access

detector_params = data["detector"]
bias_params = data["bias"]

logger.info("User parameters loaded successfully.")