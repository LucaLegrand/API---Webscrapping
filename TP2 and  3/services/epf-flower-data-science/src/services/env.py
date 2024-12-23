from dotenv import load_dotenv
import os


def load_environment():
    dotenv_path = os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)