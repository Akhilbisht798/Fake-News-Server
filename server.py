from flask import Flask, request, jsonify
import joblib
import logging

app = Flask(__name__)
vectorizer, model = joblib.load('svm_model.pkl')
logging.basicConfig(level=logging.DEBUG)


@app.route("/")
def hello_world():
    return "<h2>This is Basic Web Server Using Flask for Machine Learing.</h2>"


@app.route("/Predict/", methods=['POST', 'GET'])
def predict_text():
    try:
        data = request.get_json()
        text = data['text']
        text = [text]
        vector_text = vectorizer.transform(text)
        predict = model.predict(vector_text)
        result = ""
        if predict[0] == 0:
            result = "FAKE"
        else:
            result = "TRUE"
        response = {"message": f'Predicted: {result}'}
        return jsonify(response), 200
    except:
        return jsonify({'error': "Invalid Json data"}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
