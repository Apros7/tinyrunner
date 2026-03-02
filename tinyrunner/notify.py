import json, urllib.request

NOTIFY_URL = "http://localhost:8151/notify"
NOTIFY_SECRET = "217cabab-791c-4432-b6ba-474fa1c08718"

def notify(text: str) -> bool:
  try:
    req = urllib.request.Request(NOTIFY_URL, method="POST",
      data=json.dumps({"secret": NOTIFY_SECRET, "text": text}).encode(),
      headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=5)
    return True
  except Exception:
    return False
