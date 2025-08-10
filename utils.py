import os
from dotenv import load_dotenv

def reload_env():
    load_dotenv(override=True)
    # Optionally, re-apply env vars to any config objects here if you use them