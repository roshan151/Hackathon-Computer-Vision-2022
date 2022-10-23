import streamlit as st
from PIL import Image, ImageOps
import os
import pandas as pd
import requests
import json
import cv2
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
    bounds = result['geometry']['bounds']
    counter = 0
    lat = 0
    lng = 0
    for i in bounds.values():
        lat = lat + i['lat']
        lng = lng + i['lng']
        counter = counter + 1
    norm_lat = round(lat/counter,7)
    norm_lng = round(lng/counter,7)
    sachin_api_key = "AIzaSyCzkAF2HvnKpg0TpWPTH1TS9ZzmEWP1Mns"
    index = 1
    center = f"{norm_lat},{norm_lng}"
    zoom = 19
    maptype = "satellite"
    r = requests.get(f"https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom={zoom}&format=jpg&markers=size:mid%7Ccolor:red:S%7C{norm_lat},{norm_lng}&size=400x400&maptype={maptype}&key={sachin_api_key}")
# wb mode is stand for write binary mode
    filedir='/Users/jwei4/Documents/Github/hackathon-spacepenguins'
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
    
    base_model = InceptionV3(
    input_shape = ( 400, 400, 3), 
    weights='imagenet', 
    include_top=False)


    x = layers.Flatten()(base_model.output)

    x = Dense(1024, activation='relu')(x)

    prediction = Dense(1, activation='linear')(x)

    inception_model = Model(inputs=base_model.input, outputs=prediction)

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-9)

# compile the model (should be done *after* setting layers to non trainable)
    inception_model.compile(
        optimizer=optimizer,#'rmsprop' , 
        loss='mean_absolute_error', 
        metrics = [tf.keras.metrics.MeanSquaredError(name='mean_squared_error', dtype=None),
               tf.keras.metrics.MeanAbsolutePercentageError(name='mean_absolute_percentage_error', dtype=None)],
    ) #'rmsprop'
    
    checkpoint_path = "InceptionV3_checkpoints/v2.0.1/checkpoint.hdf5"

    inception_model.load_weights( checkpoint_path )

    st.write('---')









    st.header('Predicted Price by XGboost Regression')
    xgb_model_loaded = pickle.load(open('xgb_reg.pkl', "rb"))
    prediction_test = pd.DataFrame()
    prediction_test['calculatedfinishedsquarefeet'] = [float(sqr_feet)/100]
    prediction_test['yearbuilt'] = [(int(year_built)-1900)/10]
    prediction_test['Closest Beach'] = [distance_beach]
    prediction_test['Closest Fancy Restaurant']= [distance_restaurant]
    prediction_xgb = xgb_model_loaded.predict(prediction_test)
    #print(float(sqr_feet),int(year_built),distance_beach,distance_restaurant)
    st.header('$' + str(round(prediction_xgb[0]*10000)))
    st.write('---')











    st.header('Predicted Price by Computer Vision Neural Network')
    im = Image.open(f'{filedir}/pic_1.jpg')
    im = np.array(im,dtype=np.float32)
    # st.image(image,use_column_width=True)
    # test_x = np.asarray(test_x)
    # print(test_x)
    # test_processed_x = test_x.astype("float")/255.0
    results = inception_model.predict(np.expand_dims(im/255,0), batch_size=1)
    st.header('$'+ str(round(results[0][0])))
    st.write('---')


    st.header('Predicted Price by Hybrid Model')
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    hybrid_img = image.load_img('img_path', target_size=(224, 224))
    hybrid_img_array = image.img_to_array(hybrid_img)
    hybrid_img_batch = np.expand_dims(hybrid_img_array, axis=0)
    hybrid_img_preprocessed = preprocess_input(hybrid_img_batch)
    hybrid_model = tf.keras.applications.resnet50.ResNet50()
    hybrid_model.load_weights("filenpath")
    hybrid_prediction = hybrid_model.predict(hybrid_img_preprocessed)
    
    st.header('$'+str(round(decode_predictions(hybrid_prediction)[0])))
    st.balloons()