from flask import Flask, request, jsonify, render_template
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Flask app
server = Flask(__name__)

# Dash app
dash_app = Dash(
    __name__,
    server=server,
    routes_pathname_prefix="/dashboard/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Load the pre-trained model (replace with your actual model path)
model = joblib.load("diabetes_model.pkl") 

# Load dataset for visualization
df_viz = pd.read_csv("diabetes_data.csv")

# Bar Plot
# Filter for only people with diabetes
df_diabetic = df_viz[df_viz["Diabetes_binary"] == 1]

# Choose binary columns to analyze (excluding target column)
binary_cols = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"
]

# Prepare data for percentage calculation
percent_data = []

for col in binary_cols:
    total = len(df_diabetic)
    true_count = df_diabetic[col].sum()
    false_count = total - true_count
    percent_data.append({"Feature": col, "Value": "1 (True)", "Percentage": (true_count / total) * 100})
    percent_data.append({"Feature": col, "Value": "0 (False)", "Percentage": (false_count / total) * 100})

percent_df = pd.DataFrame(percent_data)

# Plot stacked bar chart
bar_fig = px.bar(
    percent_df,
    x="Feature",
    y="Percentage",
    color="Value",
    barmode="stack",  # <-- Changed from 'group' to 'stack'
    title="Percentage of Diabetic People with True/False Feature Values"
)

# Heatmap
# Define bins and labels for grouping
df_viz['BMI_Group'] = pd.cut(df_viz['BMI'], bins=[0, 18.5, 25, 30, 35, 40, 100],
                              labels=["Underweight", "Normal", "Overweight", "Obese I", "Obese II", "Extreme"])
df_viz['Income_Group'] = pd.cut(df_viz['Income'], bins=5)

# Calculate proportion of Diabetes_binary == 1 per group
grouped = df_viz.groupby(['BMI_Group', 'Income_Group'])['Diabetes_binary'].mean().unstack().fillna(0)

# Create annotated heatmap
heatmap_fig = ff.create_annotated_heatmap(
    z=grouped.values,
    x=[str(col) for col in grouped.columns],
    y=[str(row) for row in grouped.index],
    annotation_text=[[f"{val:.2f}" for val in row] for row in grouped.values],
    colorscale="Viridis",
    showscale=True
)

# Dash layout with navbar and graphs
dash_app.layout = html.Div([
    # Navbar
    html.Nav(
        children=[
            html.Div(
                children=[
                    html.A("Diabetes Prediction", href="/", className="navbar-brand"),
                    html.Ul(
                        children=[
                            html.Li(html.A("Assess", href="/"), className="nav-item"),
                            html.Li(html.A("Dashboard", href="/dashboard"), className="nav-item")
                        ],
                        className="navbar-nav mr-auto"
                    )
                ],
                className="navbar navbar-expand-lg navbar-dark bg-dark"
            )
        ]
    ),
    
    # Main content
    html.Div([
        html.H2("Diabetes Data Dashboard"),
        dcc.Graph(figure=bar_fig),
        dcc.Graph(figure=heatmap_fig)
    ])
])

# Endpoint to serve your form page (Flask index)
@server.route('/')
def index():
    return render_template('index.html')

@server.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Ensure all required keys are present
    required_keys = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
    ]

    if not all(key in data for key in required_keys):
        return jsonify({"error": "Missing data"}), 400

    # Convert input values to floats
    features = np.array([float(data[key]) for key in required_keys]).reshape(1, -1)

    # Make prediction
    probability = model.predict_proba(features)[0][1]
    risk_percent = round(probability * 100, 2)


    if risk_percent >= 70:
        risk_level = "High risk"
    elif risk_percent >= 40:
        risk_level = "Moderate risk"
    else:
        risk_level = "Low risk"

    return jsonify({
    "risk_percent": risk_percent,
    "risk_level": risk_level,
    "message": f"Estimated diabetes risk is {risk_level} ({risk_percent}%)"
    })

if __name__ == "__main__":
    server.run(debug=True, host="0.0.0.0", port=8000)
