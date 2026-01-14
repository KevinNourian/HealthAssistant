from dotenv import load_dotenv
import os
from serpapi import GoogleSearch  # Works with "google-search-results" package

# Load key
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY not found in .env")

params = {
    "q": "Latest COVID-19 variants 2026",
    "engine": "google",
    "api_key": SERPAPI_API_KEY,
}

search = GoogleSearch(params)
result = search.get_dict()

# Print top organic results if present
if "organic_results" in result:
    for i, item in enumerate(result["organic_results"][:3]):
        print(f"{i+1}. {item.get('title')}")
        print(f"   {item.get('snippet')}\n")
else:
    print("No results found. Check your API key or query.")
