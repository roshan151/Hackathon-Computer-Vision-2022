import streamlit as st
from PIL import Image, ImageOps
import os
import pandas as pd
import requests
import json
from PIL import Image 
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pickle
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


my_api = 'AIzaSyB1EaMZJOmFb11I9f4sBwHtbwHzVgw_UXI'
st.write("""# Housing Price Prediction App""")
st.write('---')
image=Image.open('title_image.png')
st.image(image,use_column_width=True)
st.sidebar.header('User Input Features')
address=st.sidebar.text_area("Input Address")
year_built=st.sidebar.text_area("Input Year Built")
sqr_feet=st.sidebar.text_area("Input Square Feet")

def restaurant_info(latitude, longitude):
    radius = 1000
    
    closest_restaurant = 'Not found'
    closest_restaurant_distance = 100
    try:
        #restaurants near me
        query = 'Restaurants Near Me'
        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&location={latitude},{longitude}&radius={radius}&region=us&key={my_api}"
        response = requests.request("GET", url, headers={})
        search_response = json.loads(response.content)
        search_results = search_response['results']
        counter = 0
        
        for value in search_results:
            counter = counter + 1
            if counter >= 10:
                break
            if 'name' in value.keys():
                restaurant_name = value['name']
            else:
                restaurant_name = value['formatted_address']
                
            address = value['formatted_address']
            if 'price_level' in value.keys() and value['price_level'] == 3:
                url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={latitude},{longitude}&destinations={address}&region=us&key={my_api}"
                response = requests.request("GET", url, headers={})
                distance_response = json.loads(response.content)
                distance = distance_response['rows'][0]['elements'][0]['distance']['text']
                
                if closest_restaurant_distance > float(distance.split(' ')[0]):
                    closest_restaurant_distance = float(distance.split(' ')[0])
                    closest_restaurant = restaurant_name
                    
            elif 'price_level' in value.keys() and value['price_level'] == 2:
                if value['rating'] >= 4.5 or value['user_ratings_total'] > 300:
                    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={latitude},{longitude}&destinations={address}&region=us&key={my_api}"
                    response = requests.request("GET", url, headers={})
                    distance_response = json.loads(response.content)
                    distance = distance_response['rows'][0]['elements'][0]['distance']['text']
                    
                    if closest_restaurant_distance > float(distance.split(' ')[0]):
                        closest_restaurant_distance = float(distance.split(' ')[0])
                        closest_restaurant = restaurant_name
                
        return closest_restaurant_distance, closest_restaurant
    except:
        return closest_restaurant_distance, closest_restaurant
    
def closest_beach_info(latitude, longitude):
    my_api = 'AIzaSyB1EaMZJOmFb11I9f4sBwHtbwHzVgw_UXI'
    radius = 1000

    closest_beach = 'Not Found'
    closest_beach_distance = 100
    
    try:
        query = 'Beaches Near Me'
        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&location={latitude},{longitude}&radius={radius}&region=us&key={my_api}"
        response = requests.request("GET", url, headers={})
        search_response = json.loads(response.content)
        search_results = search_response['results']
        check = False
        counter = 0

        for value in search_results:

            counter = counter + 1

            if counter >= 5:
                break
            if 'name' in value.keys():
                beach_name = value['name']
            else:
                beach_name = value['formatted_address']
            address = value['formatted_address']
            url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={latitude},{longitude}&destinations={address}&region=us&key={my_api}"
            response = requests.request("GET", url, headers={})
            distance_response = json.loads(response.content)
            distance = distance_response['rows'][0]['elements'][0]['distance']['text']
            if closest_beach_distance > float(distance.split(' ')[0]):
                closest_beach_distance = float(distance.split(' ')[0])
                closest_beach = beach_name



        return closest_beach_distance, closest_beach
    except:
        return closest_beach_distance, closest_beach

if st.button('Get result'):
    address_split = address.split(' ')
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={my_api}'
    response = requests.request("GET", url, headers={})
    search_response = json.loads(response.content)
    search_results = search_response['results']
    result = search_results[0]
    print(result)
    geometry = result['geometry']
    if 'location' in geometry.keys():
        norm_lat = geometry['location']['lat']
        norm_lng = geometry['location']['lng']
        
    elif 'bounds' in geometry.keys():
        counter = 0
        lat = 0
        lng = 0
        for i in bounds.values():
            lat = lat + i['lat']
            lng = lng + i['lng']
            counter = counter + 1
        norm_lat = round(lat/counter,7)
        norm_lng = round(lng/counter,7)
    else:
        st.write('Coordinates not found')

    sachin_api_key = "AIzaSyCzkAF2HvnKpg0TpWPTH1TS9ZzmEWP1Mns"
    index = 1
    center = f"{norm_lat},{norm_lng}"
    zoom = 19
    maptype = "satellite"
    r = requests.get(f"https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom={zoom}&format=jpg&markers=size:mid%7Ccolor:red:S%7C{norm_lat},{norm_lng}&size=400x400&maptype={maptype}&key={sachin_api_key}")
# wb mode is stand for write binary mode
    filedir='C:/Users/RTiwari1/hackathon-spacepenguins'
    f = open(f'{filedir}/pic_{index}.jpg', 'wb')
    f.write(r.content)
    f.close() 
    st.header('Satellite Image for the Address')
    address
    si=st.image('pic_1.jpg',use_column_width=True)
    distance_beach, beach_name = closest_beach_info(norm_lat, norm_lng)
    distance_restaurant, restaurant_name = restaurant_info(norm_lat, norm_lng)
    if beach_name != 'Not found':
        st.write(f'***{distance_beach} km from {beach_name}.***' )
    else:
        st.write('***No beach found nearby.***')
    if restaurant_name != 'Not found':
        st.write(f'***{distance_restaurant} km from price level 3 restaurant {restaurant_name}.***')
    else:
        st.write('***No price level 3 restaurant nearby.***')

    st.header('Predicted Price by XGboost Regression')
    xgb_model_loaded = pickle.load(open('xgb_reg.pkl', "rb"))
    prediction_test = pd.DataFrame()
    prediction_test['calculatedfinishedsquarefeet'] = [float(sqr_feet)/100]
    prediction_test['yearbuilt'] = [(int(year_built)-1900)/10]
    prediction_test['Closest Beach'] = [distance_beach]
    prediction_test['Closest Fancy Restaurant']= [distance_restaurant]
    prediction_xgb = xgb_model_loaded.predict(prediction_test)
    #print(float(sqr_feet),int(year_built),distance_beach,distance_restaurant)
    st.write('$' + str(round(prediction_xgb[0]*10000)))
    st.write('---')


    st.header('Predicted Price by Computer Vision Neural Network')
    st.write('---')