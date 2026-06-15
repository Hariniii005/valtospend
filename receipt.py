"""
receipt.py — AI-powered receipt scanner for ValtoSpend
Uses Claude AI (Anthropic API) to extract spending details from receipt photos.
"""
import base64
import json
import urllib.request


def read_receipt_with_ai(image_bytes, media_type="image/jpeg"):
    """
    Send a receipt image to Claude AI and extract:
    - amount (total spent)
    - category (Food, Groceries, Transport, etc.)
    - note (short description of purchase)

    Returns a dict: {"amount": float, "category": str, "note": str}
    or {"error": str} if something goes wrong.
    """
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 300,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8")
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Look at this receipt image. Extract the total amount spent and "
                        "the most appropriate spending category from this list: "
                        "Food, Groceries, Transport, Entertainment, Shopping, Rent, Bills, Healthcare, Education. "
                        "Also write a short note describing what was purchased. "
                        "Respond ONLY with valid JSON, no extra text, in this exact format: "
                        '{"amount": 12.50, "category": "Food", "note": "Lunch at cafe"}'
                    )
                }
            ]
        }]
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = result["content"][0]["text"].strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
    except Exception as e:
        return {"error": str(e)}
