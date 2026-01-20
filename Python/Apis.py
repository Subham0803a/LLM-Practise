# import asyncio

# async def hello():
#     print("Hello, World!")
#     await asyncio.sleep(2)
#     print("Hello again after 2 seconds!")
    
# asyncio.run(hello())

# -----------------------------------------------------

# import asyncio
# import httpx

# async def fetch_data():
#     url = "https://randomuser.me/api/"
#     async with httpx.AsyncClient() as client:
#         response = await client.get(url)
#         data = response.json()

#         user = data["results"][0]
#         print("Country:", user["location"]["country"])
#         print("Name:", user["name"]["first"], user["name"]["last"])
#         print("Age:", user["dob"]["age"])

# asyncio.run(fetch_data())
    
# -----------------------------------------------------

# import asyncio
# import httpx

# async def get_location_data(city):
#     url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    
#     async with httpx.AsyncClient() as client:
#         response = await client.get(url)
#         data = response.json()
        
#         if "results" not in data:
#             return "City not found"
        
#         return data

# city_name = input("Enter city name: ")
# location_data = asyncio.run(get_location_data(city_name))

# print(f"Location data for {city_name}:-", location_data)

# -----------------------------------------------------

# import asyncio
# import httpx

# async def get_location(city):
#     url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"

#     async with httpx.AsyncClient() as client:
#         response = await client.get(url)
#         data = response.json()

#         # if "results" not in data:
#         #     return "City not found"

#         location = data["results"][0]
#         return location["latitude"], location["longitude"], location["name"], location["country"]


# async def get_weather(lat, lon):
#     url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

#     async with httpx.AsyncClient() as client:
#         response = await client.get(url)
#         data = response.json()

#         return data["current_weather"]


# async def main():
#     city = input("Enter your city/state name: ")

#     location = await get_location(city)

#     if not location:
#         print("Location not found!")
#         return

#     lat, lon, name, country = location

#     weather = await get_weather(lat, lon)

#     print("\n🌍 Location:", name, ",", country)
#     print("📍 Latitude:", lat)
#     print("📍 Longitude:", lon)
#     print("🌡️ Temperature:", weather["temperature"], "°C")
#     print("💨 Wind Speed:", weather["windspeed"], "km/h")
#     print("🧭 Wind Direction:", weather["winddirection"])
#     print("⏰ Time:", weather["time"])


# asyncio.run(main())
