from . import db

class PredictionHistory(db.Model):
    id= db.Column(db.Integer, primary_key=True)
    year= db.Column(db.String(5))
    gdp= db.Column(db.String(150))
    imp= db.Column(db.String(150))
    crop= db.Column(db.String(150))
    prediction= db.Column(db.String(200))

