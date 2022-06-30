# Import Libraries
from flask import Flask,request, jsonify,render_template
import numpy as np
import pickle


# Load Model
Model = pickle.load(open('carAcc.sav', 'rb'))

# Create server
app = Flask(__name__)

#Routing
@app.route('/')
def index():
    return "Hello"

@app.route('/predict/')
def RoadsClassify():
    MTFC = float(request.args.get('MTFC'))
    Start_Time = float(request.args.get('Start_Time'))
    Time_Interval = float(request.args.get('Time_Interval'))
    Temperature_F_ = float(request.args.get('Temperature_F_'))
    Wind_Chill_F_ = float(request.args.get('Wind_Chill_F_'))
    Humidity___ = float(request.args.get('Humidity___'))
    Pressure_in_ = float(request.args.get('Pressure_in_'))
    Visibility_mi_ = float(request.args.get('Visibility_mi_'))
    Wind_Speed_mph_ = float(request.args.get('Wind_Speed_mph_'))
    Sunrise_Sunset = float(request.args.get('Sunrise_Sunset'))
    Shape_Length = float(request.args.get('Shape_Length'))
    Eculidian_length = float(request.args.get('Eculidian_length'))
    Sinusity = float(request.args.get('Sinusity'))
    Road_Category = float(request.args.get('Road_Category'))
    Weather_Condition_Clear = float(request.args.get('Weather_Condition_Clear'))
    Weather_Condition_Cloudy = float(request.args.get('Weather_Condition_Cloudy'))
    Weather_Condition_Fog = float(request.args.get('Weather_Condition_Fog'))
    Weather_Condition_Heavy_Rain = float(request.args.get('Weather_Condition_Heavy_Rain'))
    Weather_Condition_Rainy = float(request.args.get('Weather_Condition_Rainy'))
    Weather_Condition_Snowing = float(request.args.get('Weather_Condition_Snowing'))
    Weather_Condition_Stormy = float(request.args.get('Weather_Condition_Stormy'))
    Weather_Condition_Windy = float(request.args.get('Weather_Condition_Windy'))
    Weather_Condition_Thunder_Storm = float(request.args.get('Weather_Condition_Thunder_Storm'))
    #working with requetst
    pred = Model.predict(np.array([MTFC, Start_Time, Time_Interval, Temperature_F_, Wind_Chill_F_, Humidity___, Pressure_in_,
                                   Visibility_mi_, Wind_Speed_mph_, Sunrise_Sunset, Shape_Length, Eculidian_length, Sinusity, 
                                   Road_Category, Weather_Condition_Clear, Weather_Condition_Cloudy, Weather_Condition_Fog, 
                                   Weather_Condition_Heavy_Rain, Weather_Condition_Rainy, Weather_Condition_Snowing, 
                                   Weather_Condition_Stormy, Weather_Condition_Thunder_Storm, Weather_Condition_Windy
    ]).reshape(1, -1)
    )

    return str(pred[0])


# http://127.0.0.1:5000/predict/?MTFC=1&Start_Time=1&Time_Interval=1&Temperature_F_=1&Wind_Chill_F_=1&Humidity___=1&Pressure_in_=1&Visibility_mi_=1&Wind_Speed_mph_=1&Sunrise_Sunset=20&Shape_Length=1&Eculidian_length=1&Sinusity=1&Road_Category=1&Weather_Condition_Clear=1&Weather_Condition_Cloudy=1&Weather_Condition_Fog=1&Weather_Condition_Heavy_Rain=1&Weather_Condition_Rainy=1&Weather_Condition_Snowing=1&Weather_Condition_Stormy=1&Weather_Condition_Thunder_Storm=1&Weather_Condition_Windy=1

if __name__ == '__main__':
    app.run(debug=True)