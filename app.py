from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

<<<<<<< Updated upstream
=======
dummy_data = pd.read_csv("Synthetic_Forest_fire_dataset.csv")
>>>>>>> Stashed changes

@app.route('/')
def hello_world():
    return render_template("forest_fire.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if output > str(0.5):
        prediction_class = 'prediction-danger'
    else:
        prediction_class = 'prediction-safe'
        
    return render_template('forest_fire.html', pred='Your Forest is in Danger.\nProbability of fire occurring is {}'.format(output), prediction_class=prediction_class)



if __name__ == '__main__':
    app.run(debug=True)
