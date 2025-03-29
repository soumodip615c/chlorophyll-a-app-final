from flask import Flask, render_template, send_file
from model_utils import run_prediction_and_plot

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_model')
def run_model():
    run_prediction_and_plot()
    return render_template('index.html', image_generated=True)

@app.route('/static/plot.png')
def display_plot():
    return send_file('static/plot.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
