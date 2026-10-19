"""Google transport; credentials are never included in saved requests or errors."""
import os
import time
import requests
from settings import PROJECT

GEMINI_MODEL = 'gemini-2.5-flash'
ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'

def get_api_key():
    key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not key:
        raise RuntimeError('Λείπει το κλειδί Google API.')
    return key

def call_google(payload, key):
    for attempt in range(3):
        try:
            response = requests.post(ENDPOINT.format(model=GEMINI_MODEL),
                headers={'x-goog-api-key': key, 'Content-Type': 'application/json'},
                json=payload, timeout=(15, 180))
        except requests.RequestException:
            raise RuntimeError('Αποτυχία σύνδεσης με το Google API.') from None
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(5 * (attempt + 1))
            continue
        if not response.ok:
            raise RuntimeError(f'Google API HTTP {response.status_code}.')
        return response.json()
    raise RuntimeError('Δεν ολοκληρώθηκε η κλήση Google.')
