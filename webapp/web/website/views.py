# # views.py
from flask import Blueprint, render_template,request
from . import db
from .model import PredictionHistory
from .library import run, get_pred_string, generate_prediction_graph 

views = Blueprint('views', __name__)

@views.route('/')
def home():
    return render_template("home.html")


@views.route('/prediction', methods=['GET', 'POST'])
def prediction():
    graph_url = None
    prediction_value = None
    status_color = 'white'  # Default color
    status_message = ''
    difference = 100  # A default value, assuming 100% difference



    if request.method == "POST":
        year = request.form.get('year')
        GDP = request.form.get('GDP')
        imp = request.form.get('import')
        crop = request.form.get('crop')

        # Validate inputs
        if not all([year, GDP, imp, crop]):
            return render_template("prediction.html", error="All fields are required.")

        # Run your prediction model
        try:
            df_features, df_target, last_known_values, pred, year = run(year, GDP, imp, crop)
            prediction_value = get_pred_string(pred, year)
            graph_url = generate_prediction_graph(df_features, df_target, last_known_values, pred)


            # Calculate the difference from the goal
            pred_value= pred.flatten()[-1]
            difference = abs(0 - float(pred_value))  # Assuming pred is the percentage
            if difference <= 2:
                status_color = 'green'
                status_message = "Very close to reach the UN's zero per cent level by 2030"
            elif difference <= 5:
                status_color = 'yellow'
                status_message = "On the way, but more effort needed to reach the UN's zero per cent level by 2030"
            else:
                status_color = 'red'
                status_message = "Significant effort required to reach the UN's zero per cent level by 2030"




        except ValueError as e:
            # Handle the ValueError if conversion to float fails
            return render_template("prediction.html", error=str(e))

        # Store prediction history
        new_history = PredictionHistory(year=year, gdp=GDP, imp=imp, crop=crop, prediction=prediction_value)
        db.session.add(new_history)
        db.session.commit()


        return render_template("prediction.html", graph_url=graph_url, prediction_value=prediction_value, status_color=status_color, status_message=status_message, difference=difference)
        # return render_template("prediction.html", graph_url=graph_url, prediction_value=prediction_value)

    return render_template("prediction.html")


@views.route('/history')
def history():
    history_records = PredictionHistory.query.all()
    return render_template("history.html", history=history_records)


