# Road Image Classification

This repository contains code and resources for performing Exploratory Data Analysis (EDA), training a Convolutional Neural Network (CNN) model, and deploying the model using Flask for the classification of road images into two categories: **pothole** and **normal**.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [CNN Model Training](#cnn-model-training)
- [Model Deployment using Flask](#model-deployment-using-flask)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Road condition assessment is crucial for infrastructure maintenance and safety. This project aims to classify road images into two categories: potholes and normal road conditions. We achieve this through a combination of data analysis, deep learning, and web application development.

## Dataset
The dataset used for this project consists of road images divided into two categories: pothole and normal. You can download the dataset from [link to dataset repository or source]. The dataset is structured as follows:
```
dataset/
    ├── pothole/
    │   ├── pothole_1.jpg
    │   ├── pothole_2.jpg
    │   └── ...
    ├── normal/
    │   ├── normal_1.jpg
    │   ├── normal_2.jpg
    │   └── ...
```

## Exploratory Data Analysis (EDA)
The EDA notebook (`pothole_analysis.ipynb`) explores the dataset to gain insights into the distribution of classes, image characteristics, and potential data preprocessing steps. It also provides visualizations to better understand the dataset.

## CNN Model Training
The `pothole_analysis.ipynb` script contains code to preprocess the data, train a CNN model, and save the trained model weights. We use libraries like TensorFlow and Keras to build and train the deep learning model.

## Model Deployment using Flask
The model trained in the previous step can be deployed using a Flask web application. The `app.py` file in the `webapp/` directory sets up a web server and exposes an API endpoint for making predictions using the trained model. Users can upload road images through a web interface and receive predictions for the road condition.

## Usage
1. Clone this repository to your local machine.
2. Download the dataset and place it in the `dataset/` directory.
3. Run the EDA notebook to understand the dataset.
4. Train the CNN model using `pothole_analysis.ipynb`.
5. Deploy the model using Flask by running `app.py`.
6. Access the web application at `http://localhost:5000` and use it for road image classification.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
