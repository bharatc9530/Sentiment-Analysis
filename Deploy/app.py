import config
import torch
import flask
import time
import datetime
from flask import Flask
from flask import request
from model import BERTBaseUncased
import functools
import torch.nn as nn
from flask import render_template



app = Flask(__name__)



MODEL = None
DEVICE = config.DEVICE
PREDICTION_DICT = dict()


def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['Post'] )
def predict():
    response = {}
    if request.method == 'POST':
        message = request.form['review']
        sentence = message
        start_time = time.time()
        positive_prediction = sentence_prediction(sentence)
        negative_prediction = 1 - positive_prediction
        
        response["response"] = {
	    "time": str(datetime.datetime.now())[:-7],
	    "positive": str(positive_prediction),
	    "negative": str(negative_prediction),
	    "sentence": str(sentence),
	    "time_taken": str(time.time() - start_time),
            "message":message,
        }
    #return render_template('index.html')
    return render_template('index1.html',prediction = response)
    #return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(debug=True)
    #app.run(host="0.0.0.0", port="8080", debug=True)
