import json
import random
from datetime import datetime

data = {
    "random_number": random.randint(1, 100),
    "current_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open('data.json', 'w') as f:
    json.dump(data, f)