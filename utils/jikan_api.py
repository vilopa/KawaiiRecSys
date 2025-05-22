import requests
from typing import Dict, Optional, Union

def fetch_anime_image(title: str) -> str:
    """
    Fetch the anime poster image from Jikan API.
    Returns a URL string or a placeholder if not found.
    """
    try:
        response = requests.get(
            "https://api.jikan.moe/v4/anime",
            params={"q": title, "limit": 1}
        )
        data = response.json()
        return data["data"][0]["images"]["jpg"]["large_image_url"]
    except Exception as e:
        print(f"[Jikan Error] {title}: {e}")
        return "https://via.placeholder.com/120/ff4baf/ffffff?text=No+Image"

def fetch_anime_data(title: str) -> Dict[str, Union[str, None]]:
    """
    Fetch detailed anime data from Jikan API.
    Returns a dictionary with image, synopsis, and other details.
    """
    try:
        response = requests.get(
            "https://api.jikan.moe/v4/anime",
            params={"q": title, "limit": 1}
        )
        data = response.json()
        anime = data["data"][0]
        
        return {
            "image_url": anime["images"]["jpg"]["large_image_url"],
            "synopsis": anime.get("synopsis"),
            "trailer_url": anime.get("trailer", {}).get("url"),
            "episodes": anime.get("episodes"),
            "genres": [genre["name"] for genre in anime.get("genres", [])]
        }
    except Exception as e:
        print(f"[Jikan Error] {title}: {e}")
        return {
            "image_url": "https://via.placeholder.com/120/ff4baf/ffffff?text=No+Image",
            "synopsis": None,
            "trailer_url": None,
            "episodes": None,
            "genres": None
        } 