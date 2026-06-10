from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

teams = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Rajasthan Royals",
    "Delhi Capitals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
    "Royal Challengers Bengaluru"
]

cities = [
    "Bangalore",
    "Chandigarh",
    "Delhi",
    "Mumbai",
    "Kolkata",
    "Jaipur",
    "Hyderabad",
    "Chennai",
    "Ahmedabad",
    "Kochi",
    "Pune",
    "Lucknow",
    "Guwahati",
    "Mohali",
    "Bengaluru"
]

match_types = [
    "League",
    "Qualifier 1",
    "Qualifier 2",
    "Eliminator",
    "Semi Final",
    "Final"
]

venues = sorted(encoders["venue"].classes_.tolist())


@app.route("/")
def home():
    return render_template(
        "index.html",
        teams=teams,
        cities=cities,
        match_types=match_types,
        venues=venues
    )


@app.route("/predict", methods=["POST"])
def predict():

    team1_name = request.form["team1"]
    team2_name = request.form["team2"]

    city = request.form["city"]
    venue = request.form["venue"]
    match_type = request.form["match_type"]

    season_year = int(request.form["season_year"])

    toss_winner_is_team1 = int(
        request.form["toss_winner_is_team1"]
    )

    bat_first = int(
        request.form["bat_first"]
    )

    city_encoded = encoders["city"].transform([city])[0]
    venue_encoded = encoders["venue"].transform([venue])[0]
    match_type_encoded = encoders["match_type"].transform([match_type])[0]

    team1_encoded = encoders["team1"].transform([team1_name])[0]
    team2_encoded = encoders["team2"].transform([team2_name])[0]

    features = np.array([[
        season_year,
        city_encoded,
        venue_encoded,
        match_type_encoded,
        team1_encoded,
        team2_encoded,
        toss_winner_is_team1,
        bat_first
    ]])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    team1_prob = round(probabilities[1] * 100, 2)
    team2_prob = round(probabilities[0] * 100, 2)

    winner = team1_name if prediction == 1 else team2_name

    return render_template(
        "index.html",
        teams=teams,
        cities=cities,
        match_types=match_types,
        venues=venues,
        winner=winner,
        team1_name=team1_name,
        team2_name=team2_name,
        team1_prob=team1_prob,
        team2_prob=team2_prob
    )


if __name__ == "__main__":
    app.run(debug=True)