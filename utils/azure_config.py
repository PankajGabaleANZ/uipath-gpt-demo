import os
from pathlib import Path
from dotenv import load_dotenv

def get_config():
    # Get the root directory of the project
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Load environment variables from .env file
    load_dotenv(BASE_DIR / 'venv' / '.env')

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    model = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
    apiversion = os.getenv("AZURE_OPEN_AI_VERSION")
    base_url = os.getenv("AZURE_OPENAI_SERVICE")

    config = [
        {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "api_type": "azure",
            "api_version": apiversion
        },
    ]

    return config, api_key, endpoint, apiversion, model
