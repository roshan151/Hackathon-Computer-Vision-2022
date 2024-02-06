Property price prediction is a key capability to acquire a business advantage over competitors. Our model aims to provide a lightweight, easily scalable property price predictor that is universal and readily accessible. Let us compare the difference in the satellite imagery of these two property locatiions. First, we have a high end society:

<img width="328" alt="fancy" src="https://user-images.githubusercontent.com/96445798/210300951-0425fac0-5788-414b-991c-1da822b3918f.png">


And then we have an image of less expensive property locaton:


<img width="328" alt="regular" src="https://user-images.githubusercontent.com/96445798/210301012-22061d62-695e-43af-ab1f-47a3f52a8162.png">


We can see clear differences from top view of these locations. Our goal was to train a computer vision model that is able to understand these differences and aid in property price prediction. To do so we built a hybrid model - CNN model to use image as input and XGBoost model for all numerical data like area of the property, its distance from top rated restaurants, and year of construction to come up with the final prediction. This is how the model architecture looks like:


<img width="278" alt="archi" src="https://user-images.githubusercontent.com/96445798/210301503-da1fadd0-2c6f-4c5c-8ccf-b9e5a6178241.PNG">


We used docker container to deploy the model and built a user friendly streamlit frontend. To get predictions user enters a valid address, year of property constuction, and square footage of the apartment in their respective prompt boxes. The address is used to get the coordinates of the property and then the google earth api is used to downloaded its satellite image. Model uses this image and other information from google maps api (like distance from beach or a top rated restaurant) to predict the final price.


<img width="765" alt="Picture1" src="https://user-images.githubusercontent.com/96445798/210300803-399a0ea3-df83-4e04-afd5-e80321e0ef29.png">



