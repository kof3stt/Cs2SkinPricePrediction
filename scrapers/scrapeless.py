import requests
import json

url = "https://api.scrapeless.com/api/v1/scraper/request"

payload = json.dumps({
   "actor": "scraper.google.trends",
   "input": {
      "engine": "google.trends.search",
      "data_type": "interest_over_time",
      "q": "Counter-Strike 2",
      "date": "all"
   }
})
headers = {
   'x-api-token': 'sk_AFjOxOlkX9jXd4ToiExODCMRZzA8WqGTKpjHHLbp8d1mHIF9vQOtRpIN0iwmqddg',
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
