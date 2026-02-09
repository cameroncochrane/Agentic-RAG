import json
import os
from pathlib import Path

def load_groq_key(json_path):
    """
    Load a Groq API key from a JSON file.
    Args:
        json_path (pathlib.Path | os.PathLike | str): Path to a JSON file containing
            one of the expected API key fields.
    Returns:
        str: The Groq API key converted to a string.
    Behavior:
        - Reads and parses the JSON file at json_path.
        - Looks for the first present key in this order: "api_key", "groq_api_key",
          "key", "token".
        - Returns the associated value as a string.
    Raises:
        FileNotFoundError: If the provided path does not exist.
        KeyError: If none of the expected keys are found in the JSON object.
        json.JSONDecodeError: If the file content is not valid JSON.
    """

    p = json_path

    if not p.exists():
        raise FileNotFoundError("{p} not found")

    data = json.loads(p.read_text())
    groq_api_key = data.get("api_key") or data.get("groq_api_key") or data.get("key") or data.get("token")
    if not groq_api_key:
        raise KeyError("No Groq API key found in {p} (expected 'api_key' or 'groq_api_key')")
    
    groq_api_key = str(groq_api_key)

    return groq_api_key

def load_openai_key(json_path):
    """
    Load an OpenAI API key from a JSON file.
    Args:
        json_path (pathlib.Path | os.PathLike | str): Path to a JSON file containing
            one of the expected API key fields.
    Returns:
        str: The OpenAI API key converted to a string.
    Behavior:
        - Reads and parses the JSON file at json_path.
        - Looks for the first present key in this order: "api_key", "openai_api_key",
          "key", "token".
        - Returns the associated value as a string.
    Raises:
        FileNotFoundError: If the provided path does not exist.
        KeyError: If none of the expected keys are found in the JSON object.
        json.JSONDecodeError: If the file content is not valid JSON.
    """

    p = json_path

    if not p.exists():
        raise FileNotFoundError("{p} not found")

    data = json.loads(p.read_text())
    openai_api_key = data.get("api_key") or data.get("openai_api_key") or data.get("key") or data.get("token")
    if not openai_api_key:
        raise KeyError("No OpenAI API key found in {p} (expected 'api_key' or 'openai_api_key')")
    
    openai_api_key = str(openai_api_key)

    return openai_api_key

def load_tavily_key(json_path):
    """
    Load a tavily API key from a JSON file.
    Args:
        json_path (pathlib.Path | os.PathLike | str): Path to a JSON file containing
            one of the expected API key fields.
    Returns:
        str: The tavily API key converted to a string.
    Behavior:
        - Reads and parses the JSON file at json_path.
        - Looks for the first present key in this order: "api_key", "tavily_api_key",
          "key", "token".
        - Returns the associated value as a string.
    Raises:
        FileNotFoundError: If the provided path does not exist.
        KeyError: If none of the expected keys are found in the JSON object.
        json.JSONDecodeError: If the file content is not valid JSON.
    """

    p = json_path

    if not p.exists():
        raise FileNotFoundError("{p} not found")

    data = json.loads(p.read_text())
    tavily_api_key = data.get("api_key") or data.get("tavily_api_key") or data.get("key") or data.get("token")
    if not tavily_api_key:
        raise KeyError("No tavily API key found in {p} (expected 'api_key' or 'tavily_api_key')")
    
    tavily_api_key = str(tavily_api_key)

    return tavily_api_key
