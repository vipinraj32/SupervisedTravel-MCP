from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
import requests

mcp=FastMCP(
    "Weather MCP Server",
)
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

@mcp.tool()
def get_current_weather(location:str):

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": location,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
    )

    data = response.json()

    if response.status_code != 200:
        return f"Error: {data.get('message', 'Unable to fetch weather data')}"

    return {
        "city": data["name"],
        "temperature": data["main"]["temp"],
        "feels_like_C": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"]
    }


@mcp.tool()
def get_forecast(location:str):

    url = "https://api.openweathermap.org/data/2.5/forecast"

    params = {
        "q": location,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params)

    data = response.json()

    forecast=[]

    for item in data.get("list", []):
        forecast.append({
            "datetime": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "feels_like_C": item["main"]["feels_like"],
            "humidity": item["main"]["humidity"],
            "condition": item["weather"][0]["description"],
            "wind_speed": item["wind"]["speed"]
        })


    return {
        "city":location,
        "forecast":forecast
    }




if __name__ == "__main__":
    mcp.run()