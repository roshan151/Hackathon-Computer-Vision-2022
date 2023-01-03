Property price prediction is a key capability to acquire a business advantage over competitors. Our model aims to provide a lightweight, easily scalable property price predictor that is universal and readily accessible. Let us compare the difference in the satellite imagery of these two property locatiions. First, we have a high end society:
<img width="328" alt="fancy" src="https://user-images.githubusercontent.com/96445798/210300951-0425fac0-5788-414b-991c-1da822b3918f.png">
And then we have an image of less expensive property locaton:
<img width="328" alt="regular" src="https://user-images.githubusercontent.com/96445798/210301012-22061d62-695e-43af-ab1f-47a3f52a8162.png">
We aim to train a computer vision model to be able to capture these differences and aid in price prediction. This a hybrid model -a CNN model utilizes the image and an XGBoost model takes in other numerical data like distance from top rated restaurant, area of the property and built year to come up with the final prediction. This is how the model architecture looks like:
<img width="278" alt="archi" src="https://user-images.githubusercontent.com/96445798/210301503-da1fadd0-2c6f-4c5c-8ccf-b9e5a6178241.PNG">
We deployed the model on kubernetes and built a streamlit frontend. Firts, we enter a valid addres with the age and area of apartment in the prompt box. The address is used to ping the google earth api and a satellite image is downloaded. Then the model uses this image and other information from google maps api to predict a price.
<img width="765" alt="Picture1" src="https://user-images.githubusercontent.com/96445798/210300803-399a0ea3-df83-4e04-afd5-e80321e0ef29.png">


