# File path: src/ingestion/download_data.py
import io
import os
import zipfile
import requests

# Combined list of Formats, Leagues, Years, and Countries
DATA_URLS = {
    # FORMATS & LEAGUES
    "tests": "https://cricsheet.org/downloads/tests_female_json.zip",
    "odis": "https://cricsheet.org/downloads/odis_female_json.zip",
    "it20s": "https://cricsheet.org/downloads/it20s_female_json.zip",
    "hundred": "https://cricsheet.org/downloads/hnd_female_json.zip",
    "super_smash": "https://cricsheet.org/downloads/ssm_female_json.zip",
    "womens_one_day": "https://cricsheet.org/downloads/wod_female_json.zip",
    "womens_t20_blast": "https://cricsheet.org/downloads/wtb_female_json.zip",
    "icc_world_cup": "https://cricsheet.org/downloads/icc_womens_cricket_world_cup_female_json.zip",
    "wpl": "https://cricsheet.org/downloads/wpl_json.zip",
    "wbbl": "https://cricsheet.org/downloads/wbb_json.zip",

    # YEAR WISE DATA
    "2016": "https://cricsheet.org/downloads/2016_female_json.zip",
    "2017": "https://cricsheet.org/downloads/2017_female_json.zip",
    "2018": "https://cricsheet.org/downloads/2018_female_json.zip",
    "2019": "https://cricsheet.org/downloads/2019_female_json.zip",
    "2020": "https://cricsheet.org/downloads/2020_female_json.zip", 
    "2021": "https://cricsheet.org/downloads/2021_female_json.zip",
    "2022": "https://cricsheet.org/downloads/2022_female_json.zip",
    "2023": "https://cricsheet.org/downloads/2023_female_json.zip",
    "2024": "https://cricsheet.org/downloads/2024_female_json.zip",
    "2025": "https://cricsheet.org/downloads/2025_female_json.zip",
    "2026": "https://cricsheet.org/downloads/2026_female_json.zip",

    # BY COUNTRY
    "australia": "https://cricsheet.org/downloads/australia_female_json.zip",
    "bangladesh": "https://cricsheet.org/downloads/bangladesh_female_json.zip",
    "canada": "https://cricsheet.org/downloads/canada_female_json.zip",
    "england": "https://cricsheet.org/downloads/england_female_json.zip",
    "india": "https://cricsheet.org/downloads/india_female_json.zip",
    "ireland": "https://cricsheet.org/downloads/ireland_female_json.zip",
    "malaysia": "https://cricsheet.org/downloads/malaysia_female_json.zip",
    "nepal": "https://cricsheet.org/downloads/nepal_female_json.zip",
    "netherlands": "https://cricsheet.org/downloads/netherlands_female_json.zip",
    "new_zealand": "https://cricsheet.org/downloads/new_zealand_female_json.zip",
    "pakistan": "https://cricsheet.org/downloads/pakistan_female_json.zip",
    "scotland": "https://cricsheet.org/downloads/scotland_female_json.zip",
    "south_africa": "https://cricsheet.org/downloads/south_africa_female_json.zip",
    "sri_lanka": "https://cricsheet.org/downloads/sri_lanka_female_json.zip",
    "thailand": "https://cricsheet.org/downloads/thailand_female_json.zip",
    "uganda": "https://cricsheet.org/downloads/uganda_female_json.zip",
    "uae": "https://cricsheet.org/downloads/united_arab_emirates_female_json.zip",
    "usa": "https://cricsheet.org/downloads/united_states_of_america_female_json.zip",
    "west_indies": "https://cricsheet.org/downloads/west_indies_female_json.zip",
    "zimbabwe": "https://cricsheet.org/downloads/zimbabwe_female_json.zip"
}

PEOPLE_CSV_URL = "https://cricsheet.org/register/people.csv"

RAW_JSON_DIR = os.path.join("data", "raw_json")
REGISTRY_DIR = os.path.join("data", "registry")

def setup_directories():
    os.makedirs(RAW_JSON_DIR, exist_ok=True)
    os.makedirs(REGISTRY_DIR, exist_ok=True)

def download_and_extract_zip(url: str, extract_to: str):
    print(f"Fetching: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Filter for match JSON files
            json_files = [f for f in zip_ref.namelist() if f.endswith(".json")]
            new_files_extracted = 0
            
            for f in json_files:
                target_path = os.path.join(extract_to, f)
                # DEDUPLICATION LOGIC: Skip extracting if the match file already exists
                if not os.path.exists(target_path):
                    zip_ref.extract(f, extract_to)
                    new_files_extracted += 1
                    
            skipped = len(json_files) - new_files_extracted
            print(f"Extracted {new_files_extracted} new files (Skipped {skipped} duplicates).")
            
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
    except zipfile.BadZipFile:
        print(f"Failed to unzip {url}. File might be corrupted or empty.")

def download_people_registry():
    print(f"Downloading player registry: {PEOPLE_CSV_URL}")
    response = requests.get(PEOPLE_CSV_URL)
    response.raise_for_status()
    file_path = os.path.join(REGISTRY_DIR, "people.csv")
    with open(file_path, "wb") as f:
        f.write(response.content)
    print(f"Saved registry to {file_path}")

if __name__ == "__main__":
    setup_directories()
    for name, url in DATA_URLS.items():
        print(f"\nProcessing {name}...")
        download_and_extract_zip(url, RAW_JSON_DIR)
         
    download_people_registry()
    print("\n--- All downloads complete. ---")