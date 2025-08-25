import os
from pathlib import Path
from dotenv import load_dotenv


def get_uipath_config():
    """
    Load UiPath config values from .env and return as a dictionary.

    Returns:
        dict: A dictionary containing UiPath API credentials and parameters.
    """
    # Get the root directory of the project (adjust if needed)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Load environment variables from `.env`
    load_dotenv(BASE_DIR / '.env')

    config = {
        "client_id": os.getenv("UIPATH_OAUTH_CLIENT_ID"),
        "client_secret": os.getenv("UIPATH_OAUTH_CLIENT_SECRET"),
        "account_logical_name": os.getenv("UIPATH_CLOUD_ORG_NAME"),
        "tenant_logical_name": os.getenv("UIPATH_OAUTH_TENANT"),
        "base_url": os.getenv("UIPATH_CLOUD_URL", "https://cloud.uipath.com"),
        "folder_id": os.getenv("UIPATH_FOLDER_ID"),
    }

    # Optionally validate required keys
    missing_keys = [k for k, v in config.items() if not v]
    if missing_keys:
        raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")

    return config
