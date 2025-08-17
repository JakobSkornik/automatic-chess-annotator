import os

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # If python-dotenv is not installed, skip silently
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


