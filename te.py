import requests

api_key = "b09ae973a8d835ec1d9619e4181a5427"
base_url = "http://api.openweathermap.org/data/2.5/weather?"
# speak("City name ")
print("City name: ")
# city_name = takeCommand()
complete_url = base_url + "appid=" + api_key + "&q=" + "mumbai"
response = requests.get(complete_url)
x = response.json()
print(response,x)
