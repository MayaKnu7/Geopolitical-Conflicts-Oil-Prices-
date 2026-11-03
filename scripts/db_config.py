# db_config.py
# Shared database connection for all pipeline scripts.
# Uses a single Postgres connection string (Neon free tier) from the DATABASE_URL
# environment variable, e.g. in your local .env file:
#   DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env file "
            "(postgresql://user:password@host/dbname?sslmode=require)."
        )
    return psycopg2.connect(url, connect_timeout=10)
