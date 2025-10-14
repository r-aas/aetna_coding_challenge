"""Aetna AI Engineer Coding Challenge - Movie System CLI Entry Point."""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.cli import app

if __name__ == "__main__":
    app()
