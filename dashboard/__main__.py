"""Entry point so `python -m dashboard` works (FastAPI/uvicorn app lives in __init__)."""
from . import run

if __name__ == "__main__":
    run()
