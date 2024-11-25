#!/bin/bash
#Use this for automated download and unzipping of data set 
#else use this link to manually download the dataset "https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k"

# Set the Kaggle API credentials (replace with your actual username and key)
KAGGLE_USERNAME="your_kaggle_username"
KAGGLE_KEY="your_kaggle_api_key"

# Set the URL of the dataset
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k"

# Download the dataset using curl with Kaggle API authentication
curl -L -o flickr8k.zip \
     -H "Authorization: Basic $(echo -n "$KAGGLE_USERNAME:$KAGGLE_KEY" | base64)" \
     "$DATASET_URL"

# Unzip the downloaded dataset
unzip flickr8k.zip

echo "Dataset downloaded and extracted successfully!"