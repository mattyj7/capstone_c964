from flask import Flask, request, jsonify, render_template
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
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

# Load model and data
model = joblib.load("diabetes_model.pkl")
df_viz = pd.read_csv("diabetes_data.csv")

# Feature list (used in prediction and visualization)
required_keys = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

# ----- STATIC VISUALIZATIONS -----

# Stacked bar chart
df_diabetic = df_viz[df_viz["Diabetes_binary"] == 1]
binary_cols = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"
]

percent_data = []
for col in binary_cols:
    total = len(df_diabetic)
    true_count = df_diabetic[col].sum()
    false_count = total - true_count
    percent_data.append({"Feature": col, "Value": "1 (True)", "Percentage": (true_count / total) * 100})
    percent_data.append({"Feature": col, "Value": "0 (False)", "Percentage": (false_count / total) * 100})
percent_df = pd.DataFrame(percent_data)

bar_fig = px.bar(
    percent_df,
    x="Feature",
    y="Percentage",
    color="Value",
    barmode="stack",
    title="Percentage of Diabetic People with True/False Feature Values"
)

# Static heatmap
df_viz['BMI_Group'] = pd.cut(df_viz['BMI'], bins=[0, 18.5, 25, 30, 35, 40, 100],
                              labels=["Underweight", "Normal", "Overweight", "Obese I", "Obese II", "Extreme"])
df_viz['Income_Group'] = pd.cut(df_viz['Income'], bins=5)

grouped = df_viz.groupby(['BMI_Group', 'Income_Group'])['Diabetes_binary'].mean().unstack().fillna(0)

heatmap_fig = ff.create_annotated_heatmap(
    z=grouped.values,
    x=[str(col) for col in grouped.columns],
    y=[str(row) for row in grouped.index],
    annotation_text=[[f"{val:.2f}" for val in row] for row in grouped.values],
    colorscale="Viridis",
    showscale=True
)

# ----- DASH LAYOUT -----

dash_app.layout = html.Div([
    # Navbar
    html.Nav(
        children=[
            html.Div(
                children=[
                    html.A("Diabetes Prediction", href="/", className="navbar-brand"),
                    html.Ul(
                        children=[
                            html.Li(html.A("Assess", href="/"), className="nav-item", style={"margin-right": "20px"}),
                            html.Li(html.A("Dashboard", href="/dashboard"), className="nav-item")
                        ],
                        className="navbar-nav mr-auto"
                    )
                ],
                className="navbar navbar-expand-lg navbar-dark bg-dark px-3"
            )
        ]
    ),

    # Main Content with Page Padding
    html.Div([
        html.H2("Diabetes Data Dashboard"),
        dcc.Graph(figure=bar_fig),
        dcc.Graph(figure=heatmap_fig),

        html.H4("Predictive Heatmap by Feature"),
        html.Div([
            html.Label("Select Feature X"),
            dcc.Dropdown(
                id="feature-x-dropdown",
                options=[{"label": col, "value": col} for col in required_keys],
                value="Education"
            ),
            html.Br(),
            html.Label("Select Feature Y"),
            dcc.Dropdown(
                id="feature-y-dropdown",
                options=[{"label": col, "value": col} for col in required_keys],
                value="Income"
            ),
        ], style={"width": "50%", "margin": "20px 0"}),

        dcc.Graph(id="dynamic-heatmap")
    ], style={"padding": "30px"})  # Padding around the entire page
])


# ----- DASH CALLBACK -----

@dash_app.callback(
    Output("dynamic-heatmap", "figure"),
    [Input("feature-x-dropdown", "value"),
     Input("feature-y-dropdown", "value")]
)
def update_heatmap(feature_x, feature_y):
    if feature_x == feature_y:
        return {
            "data": [],
            "layout": {
                "title": "Please select two different features."
            }
        }

    x_range = np.linspace(df_viz[feature_x].min(), df_viz[feature_x].max(), 50)
    y_range = np.linspace(df_viz[feature_y].min(), df_viz[feature_y].max(), 50)
    xx, yy = np.meshgrid(x_range, y_range)

    base = df_viz[required_keys].mean().to_frame().T
    grid = pd.DataFrame(np.repeat(base.values, xx.size, axis=0), columns=required_keys)
    grid[feature_x] = xx.ravel()
    grid[feature_y] = yy.ravel()

    # If your model needs a scaler, replace grid with scaler.transform(grid)
    # grid_scaled = scaler.transform(grid)
    grid_scaled = grid  # Assuming model can accept raw inputs

    probs = model.predict_proba(grid_scaled)[:, 1]
    probs_grid = probs.reshape(xx.shape)

    fig = px.imshow(
        probs_grid * 100,
        x=x_range,
        y=y_range,
        color_continuous_scale="viridis",
        labels={"color": "Predicted Risk (%)"},
        aspect="auto"
    )
    fig.update_layout(
        title=f"Diabetes Risk by {feature_x} and {feature_y}",
        xaxis_title=feature_x,
        yaxis_title=feature_y
    )
    return fig

# ----- FLASK ROUTES -----

@server.route('/')
def index():
    return render_template('index.html')

@server.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Missing data"}), 400

    features = np.array([float(data[key]) for key in required_keys]).reshape(1, -1)
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
