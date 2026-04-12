import os
from dotenv import load_dotenv
import requests
import logging

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

def _headers():
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

def select(tabla, filters=None, order=None, limit=None):
    url = f"{SUPABASE_URL}/rest/v1/{tabla}"
    params = {"select": "*"}
    if filters:
        params.update(filters)
    if order:
        params["order"] = order
    if limit:
        params["limit"] = str(limit)
    r = requests.get(url, headers=_headers(), params=params)
    r.raise_for_status()
    return r.json()

def insert(tabla, data):
    url = f"{SUPABASE_URL}/rest/v1/{tabla}"
    r = requests.post(url, headers=_headers(), json=data)
    r.raise_for_status()
    return r.json()

def update(tabla, data, filters):
    url = f"{SUPABASE_URL}/rest/v1/{tabla}"
    r = requests.patch(url, headers=_headers(), json=data, params=filters)
    r.raise_for_status()
    return r.json()

def get_client():
    return {"select": select, "insert": insert, "update": update}
