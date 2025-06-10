#!/usr/bin/env python3
## Need Steam API key
## Can filter to games only, DLC, or movies


import os
import json
import random
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse

def get_api_key():
    api_key = os.getenv("STEAM_API_KEY")
    if not api_key:
        raise EnvironmentError("STEAM_API_KEY environment variable is not set!")
    return api_key

STEAM_APP_LIST_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1"

def fetch_game_descriptions(num_games=50, random_state=42):
    random.seed(random_state)
    api_key = get_api_key()
    params = {
        "key": api_key,
        "max_results": 50000,
        "include_games": 1,
        "include_dlc": 0,
        "include_software": 0,
        "include_videos": 0,
        "include_hardware": 0,
    }
    response = requests.get(STEAM_APP_LIST_URL, params=params)
    response.raise_for_status()
    apps = response.json().get("response", {}).get("apps", [])

    if not apps:
        print("Warning: No apps found in Steam API response.")
        return []

    sample_count = min(len(apps), num_games * 2)
    sampled_apps = random.sample(apps, sample_count)

    games = []
    for app in tqdm(sampled_apps, desc="Fetching game details"):
        app_id, name = app["appid"], app["name"]
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        try:
            resp = requests.get(url, timeout=5).json()
            data = resp.get(str(app_id), {}).get("data", {})
            description_html = data.get("detailed_description", "")
            description = BeautifulSoup(description_html, "html.parser").get_text().strip()
            if description:
                games.append({"game_name": name, "game_description": description})
        except Exception as e:
            print(f"Error fetching {name} ({app_id}): {e}")
            continue
        if len(games) >= num_games:
            break

    return games

def main():
    parser = argparse.ArgumentParser(description="Fetch Steam games and save details to JSON.")
    parser.add_argument("--num_games", type=int, default=50, help="Number of games to fetch.")
    parser.add_argument("--output", type=str, default="steam_games.json", help="Output JSON file path.")
    args = parser.parse_args()

    games = fetch_game_descriptions(num_games=args.num_games)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(games, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(games)} game details to {args.output}")

if __name__ == "__main__":
    main()
