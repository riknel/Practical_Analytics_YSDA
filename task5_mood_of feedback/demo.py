from codecs import open
import time
import pickle
from flask import Flask, render_template, request
import pymorphy2
app = Flask(__name__)

with open('model_dump_log.pkl', 'rb') as output_file:
    classifier = pickle.load(output_file)
with open('vectorizer_dump.pkl', 'rb') as output_file:
    vectorizer = pickle.load(output_file)


print("Classifier is ready")

def preprocess_text(text, vectorizer):
    morph = pymorphy2.MorphAnalyzer()
    text = ' '.join([morph.parse(word)[0].normal_form for word in text.split()])
    return vectorizer.transform([text])

@app.route("/sentiment-demo", methods=["POST", "GET"])
def index_page(text="", prediction_message=""):
    if request.method == "POST":
        text = request.form["text"]

    logfile = open("logs.txt", "a", "utf-8")

    logfile.write("<response>")
    logfile.write(text)
    if text == "":
        return render_template('hello.html', text=text, prediction_message="you need to write a response")

    prediction = classifier.predict(preprocess_text(text, vectorizer))
    if prediction:
        prediction_message = 'positive'
    else:
        prediction_message = 'negative'


    logfile.write(prediction_message)
    logfile.write('</response>')
    logfile.close()
	
    return render_template('hello.html', text=text, prediction_message=prediction_message)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=82, debug=False)
