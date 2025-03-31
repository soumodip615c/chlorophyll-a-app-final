from flask import Flask, render_template
from model_utils import run_prediction_and_plot

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    run_prediction_and_plot()
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)

