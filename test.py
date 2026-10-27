from tools.tavily_tool import tavily_search

from tools.flight_tool import search_flights
from backend import run_travel_agent
# response = tavily_search("Best Hotels in Bhopal India?")
# print(response)

# response = search_flights("Plan a 7 days Japan trip from Bangladesh")
# print(response)

# user_input=input("enter travel request:")

# response =run_travel_agent(
#     user_input=user_input,
#     thread_id="test_user"
# )

# print("\n Final Response:\n")
# print(response["answer"])


import asyncio
from mcp_client import get_all_tools

if __name__ == "__main__":
    asyncio.run(get_all_tools())