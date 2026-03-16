"""
JSON Serialization Helper
Fixes datetime serialization errors
"""
import json
from datetime import datetime, date, time
from decimal import Decimal

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, time):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

def safe_json_dumps(data):
    """Safely convert data to JSON, handling datetime objects"""
    return json.dumps(data, cls=DateTimeEncoder)

def clean_for_json(data):
    """Clean data structure to remove non-serializable objects"""
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(item) for item in data]
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, time):
        return data.isoformat()
    elif isinstance(data, Decimal):
        return float(data)
    else:
        return data
