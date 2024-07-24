from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Make prediction
    prediction = model.predict_proba(final_features)
    probability = prediction[0][1]  # Probability of needing treatment

    # Format the probability to two decimal places
    formatted_prob = '{:.2f}'.format(probability)

    # Determine the prediction message
    if probability > 0.5:
        result = 'You need treatment. Probability of mental illness is {}'.format(formatted_prob)
    else:
        result = 'You do not need treatment. Probability of mental illness is {}'.format(formatted_prob)

    # Render the result to the index.html template
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
