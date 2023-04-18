import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl","rb"))
@flask_app.route("/")
def Home():
    return render_template("index.html", show_fake="hidden", show_real="hidden")

@flask_app.route("/predict", methods = ["POST"])
# def predict():
#     float_features = [float(x) for x in request.form.values()]
#     features = [np.array(float_features)]
#     prediction = model.predict(features)
#     return render_template("index.html", prediction_text = "The News is {} %".format(prediction))


def predict():
    float_features = [x for x in request.form.values()]
    float_features = [float_features[1]]
    features = np.array(float_features)
    print(features.shape)
    #Transform the input
    # Initialize a TfidfVectorizer
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    features = vectorizer.transform(features)

    prediction = model.predict(features)
    if(prediction[0]=="FAKE"):
        show_real = "hidden"
        show_fake = ""

    else:
        show_real = ""
        show_fake = "hidden"


    return render_template("index.html", prediction_text = "This news is {} ".format(prediction[0]), show_real=show_real, show_fake=show_fake)

if __name__ == "__main__":
    flask_app.run(debug=True)

