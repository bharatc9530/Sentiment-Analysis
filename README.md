# Sentimental Analysis Using Bert Transformer model


```
Project workflow
├── config.py (It contain all parameter of Bert model and path for dataset) 
│    ├── IMDB Dataset.csv (It is Amazon review dataset contain review and sentiment associate with it)
│    ├── dataset.py (Load dataset, preprocessing and input for model)
│    ├── model.py (It load pretained model over dataset)
│    ├── engine.py (It contain Bert model)
│    ├── train.py (It load train, eval function for training and run it for training model)
│    └── Train & Predict.ipynb (It run all the python scripts for training and prediction in Jupyter Notebook)


```
#### Aim
Build End-to-End Machine learning pipline for preprocessing, exploratory data analysis, modelling, deployment. 

#### Dataset
Train over Amazon review dataset using Bert Transformer model and use f1-score for evalution of model. 
[Dataset](https://www.kaggle.com/bittlingmayer/amazonreviews)

#### Save model
Trained model can be download 
[Model](https://drive.google.com/file/d/10AOBLpnIStJrgtq9yH25XEG6Ml1nCA4h/view?usp=sharing)

#### Tools:
Flask, Transformers, pytorch, HTML, CSS, Javascripts, AWS EC2, nltk

#### Prediction
Deployment code is available in deploy branch containing web application integrated with Flask app
[Deploy](https://github.com/bharatc9530/Sentiment-Analysis/tree/deploy)


#### Result 
![WhatsApp Video 2020-10-18 at 4 36 06 PM](https://user-images.githubusercontent.com/58046531/96370690-54b99f00-117c-11eb-8f74-b06007d3ddb1.gif)
