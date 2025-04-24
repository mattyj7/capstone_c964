from flask import Flask, request, jsonify, render_template, Response
from dash import Dash, dcc, html
from nbconvert import HTMLExporter
import nbformat
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
# model = joblib.load("diabetes_model.pkl")
model = joblib.load("modelv2.pkl")
scaler = joblib.load("scaler.pkl")
df_viz = pd.read_csv("diabetes_data.csv")

# Remove rows where BMI is greater than 50
df_viz = df_viz[df_viz['BMI'] <= 50]

# Feature list (used in prediction and visualization)
required_keys = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

# ----- STATIC VISUALIZATIONS -----

# # Stacked bar chart
# df_diabetic = df_viz[df_viz["Diabetes_binary"] == 1]
all_feature_cols = [
    "High Blood Pressure", "High Cholesterol", "Cholesterol Check", "BMI", "Smoker",
    "Stroke", "Heart Disease or Attack", "Physical Activity", "Fruit Consumption", "Vegetable Consumption",
    "Heavy Alcohol Consumption", "Any Healthcare", "No Doctor Due to Cost", "General Health", "Mental Health Days",
    "Physical Health Days", "Difficulty Walking", "Sex", "Age Group", "Annual Income", "Education Level"
]


# percent_data = []
# for col in binary_cols:
#     total = len(df_diabetic)
#     true_count = df_diabetic[col].sum()
#     false_count = total - true_count
#     percent_data.append({"Feature": col, "Value": "1 (True)", "Percentage": (true_count / total) * 100})
#     percent_data.append({"Feature": col, "Value": "0 (False)", "Percentage": (false_count / total) * 100})
# percent_df = pd.DataFrame(percent_data)

# bar_fig = px.bar(
#     percent_df,
#     x="Feature",
#     y="Percentage",
#     color="Value",
#     barmode="stack",
#     title="Percentage of Diabetic People with True/False Feature Values"
# )

# Static heatmap
df_viz['BMI_Group'] = pd.cut(df_viz['BMI'], bins=[0, 18.5, 25, 30, 35, 40, 100],
                              labels=["Underweight", "Normal", "Overweight", "Obese I", "Obese II", "Extreme"])
df_viz['Income_Group'] = pd.cut(
    df_viz['Income'], 
    bins=[0, 2, 4, 6, 8], 
    labels=["$0K–15K", "$15K–25K", "$25K–50K", "$50K+"]
)

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

featureMap = {
    "High Blood Pressure": "HighBP",
    "High Cholesterol": "HighChol",
    "Cholesterol Check": "CholCheck",
    "BMI": "BMI",
    "Smoker": "Smoker",
    "Stroke": "Stroke",
    "Heart Disease or Attack": "HeartDiseaseorAttack",
    "Physical Activity": "PhysActivity",
    "Fruit Consumption": "Fruits",
    "Vegetable Consumption": "Veggies",
    "Heavy Alcohol Consumption": "HvyAlcoholConsump",
    "Any Healthcare": "AnyHealthcare",
    "No Doctor Due to Cost": "NoDocbcCost",
    "General Health": "GenHlth",
    "Mental Health Days": "MentHlth",
    "Physical Health Days": "PhysHlth",
    "Difficulty Walking": "DiffWalk",
    "Sex": "Sex",
    "Age Group": "Age",
    "Education Level": "Education",
    "Annual Income": "Income"
}

FEATURE_LEGEND = {
        "HighBP": "High blood pressure (1 = Yes, 0 = No)",
        "HighChol": "High cholesterol (1 = Yes, 0 = No)",
        "CholCheck": "Cholesterol check within the past 5 years (1 = Yes, 0 = No)",
        "BMI": "Body Mass Index (Continuous variable)",
        "Smoker": "Smoking status (1 = Yes, 0 = No)",
        "Stroke": "History of stroke (1 = Yes, 0 = No)",
        "HeartDiseaseorAttack": "History of heart disease or heart attack (1 = Yes, 0 = No)",
        "PhysActivity": "Physical activity in the past month (1 = Yes, 0 = No)",
        "Fruits": "1+ serving of fruits daily (1 = Yes, 0 = No)",
        "Veggies": "1+ serving of vegetables daily (1 = Yes, 0 = No)",
        "HvyAlcoholConsump": "Heavy alcohol consumption 14+ drinks a week (male) or +7 drinks a week (female) (1 = Yes, 0 = No)",
        "AnyHealthcare": "Access to healthcare (1 = Yes, 0 = No)",
        "NoDocbcCost": "No doctor's visit due to cost (1 = Yes, 0 = No)",
        "GenHlth": "Self-reported general health (1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor)",
        "MentHlth": "Mental health days in the past month (1-30)",
        "PhysHlth": "Physical health days in the past month (1-30)",
        "DiffWalk": "Difficulty walking or climbing stairs (1 = Yes, 0 = No)",
        "Sex": "Gender (1 = Male, 0 = Female)",
        "Age": "Age group (1 = 18-24, 2 = 25-29, 3 = 30-34, 4 = 35–39, 5 = 40–44, 6 = 45–49, 7 = 50–54, 8 = 55–59, 9 = 60–64, 10 = 65–69, 11 = 70–74, 12 = 75–79, 13 = 80 or older)",
        "Education": "Highest education level (1 = Less than high school, 2 = High school graduate, 3 = Some college, 4 = College graduate)",
        "Income": "Income level (1 = Less than $10,000, 2 = $10,000-$19,999, 3 = $20,000-$34,999, 4 = $35,000-$49,999, 5 = $50,000-$74,999, 6 = $75,000 or more)"
    }

dash_app.layout = html.Div([
    # Navbar
    html.Nav(
        children=[
            html.Div(
                children=[
                    html.A("Diabetes Risk Score", href="/", className="navbar-brand"),
                    html.Ul(
                        children=[
                            html.Li(html.A("Diabetes Risk Form", href="/"), className="nav-item", style={"margin-right": "20px"}),
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
        html.A("Jump to Feature Legend ↓", href="#feature-legend", style={"fontSize": "16px", "marginBottom": "20px", "display": "block"}),
        # dcc.Graph(figure=bar_fig),
        html.H4("Feature Breakdown Among Diabetics"),
        html.Div([
                html.Label("Select Feature"),
                dcc.Dropdown(
                id="bar-feature-dropdown",
                options=[{"label": col, "value": col} for col in all_feature_cols],
                value=all_feature_cols[0]
            )
        ], style={"width": "50%", "margin": "20px 0"}),

        dcc.Graph(id="dynamic-bar-chart"),

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

        dcc.Graph(id="dynamic-heatmap"),

        html.H4("Diabetes Risk Line Graph (Other Features at Mean)"),
        html.Div([
            html.Label("Select Feature"),
            dcc.Dropdown(
                id="line-feature-dropdown",
                options=[{"label": col, "value": col} for col in all_feature_cols],
                value="BMI"
            )
        ], style={"width": "50%", "margin": "20px 0"}),

        dcc.Graph(id="risk-line-graph"),
        html.H4("Income and BMI Group with Diabetic Risk Heatmap"),
        dcc.Graph(figure=heatmap_fig),
        html.Div([
            html.H4("Feature Legend", id="feature-legend"),
            html.Ul([
                html.Li(f"{feature}: {desc}") for feature, desc in FEATURE_LEGEND.items()
            ])
        ]),
    ], style={"padding": "30px"}),
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

    # Scale the grid data if necessary
    grid_scaled = grid.copy()  # Copy the grid to scale it
    grid_scaled[required_keys] = scaler.transform(grid[required_keys])  # Apply the scaling

    # Predict risk
    probs = model.predict_proba(grid_scaled)[:, 1]  # Predict probabilities
    probs_grid = probs.reshape(xx.shape)  # Reshape the probabilities to match the grid shape

    # Create the heatmap
    fig = px.imshow(
        probs_grid * 100,  # Scale the predicted risk to percentage
        x=x_range,
        y=y_range,
        color_continuous_scale="viridis",
        labels={"color": "Predicted Risk (%)"},
        aspect="auto"
    )
    fig.update_layout(
        title=f"Diabetes Risk by {feature_x} and {feature_y}",
        xaxis_title=FEATURE_LEGEND[feature_x],
        yaxis_title=feature_y
    )
    return fig

@dash_app.callback(
    Output("risk-line-graph", "figure"),
    [Input("line-feature-dropdown", "value")]
)
def update_line_graph(selected_feature):
    feature = featureMap[selected_feature]
    print(feature)
    feature_range = np.linspace(df_viz[feature].min(), df_viz[feature].max(), 200)

    # Get mean values of required features
    mean_values = df_viz[required_keys].mean().to_frame().T

    # Repeat mean values and replace selected feature with range
    base_df = pd.DataFrame(np.repeat(mean_values.values, len(feature_range), axis=0), columns=required_keys)
    base_df[feature] = feature_range

    # Scale if necessary (apply scaling here)
    scaled_df = base_df.copy()  # Copy the dataframe to scale
    
    # Assuming you have a scaler initialized, fit it if necessary (uncomment if needed)
    # scaler.fit(df_viz[required_keys])  # Fit the scaler to the training data if not done already
    
    # Apply the scaler transformation to the base_df excluding the feature column
    scaled_df[required_keys] = scaler.transform(base_df[required_keys])

    # Predict risk
    probs = model.predict_proba(scaled_df)[:, 1] * 100

    # Create figure
    fig = px.line(
        x=feature_range,
        y=probs,
        labels={"x": feature, "y": "Predicted Diabetes Risk (%)"},
        title=f"Diabetes Risk vs. {feature} (Other Features at Mean)"
    )
    fig.update_layout(
        xaxis_title=FEATURE_LEGEND[feature],
        yaxis_title="Predicted Risk (%)",
        template="plotly_white"
    )
    return fig

@dash_app.callback(
    Output("dynamic-bar-chart", "figure"),
    Input("bar-feature-dropdown", "value")
)
def update_bar_chart(selected_feature):
    feature = featureMap[selected_feature]
    df_clean = df_viz[[feature, "Diabetes_binary"]].dropna()

    # Apply readable labels for selected features
    if feature == "Sex":
        sex_map = {
                    1: "Male",
                    0: "Female"
               
        }
        df_clean[feature] = df_clean[feature].map(sex_map)
    if feature == "Education":
        education_map = {
            1: "Never attended",
            2: "Grades 1–8",
            3: "Grades 9–11",
            4: "High School Grad",
            5: "Some College/Tech",
            6: "College Grad"
        }
        df_clean[feature] = df_clean[feature].map(education_map)

    elif feature == "Income":
        income_map = {
            1: "$10K",
            2: "$10K–15K",
            3: "$15K–20K",
            4: "$20K–25K",
            5: "$25K-$35K",
            6: "$35K–50K",
            7: "$50K–75K",
            8: "≥ $75K"
        }
        df_clean[feature] = df_clean[feature].map(income_map)

    elif feature == "_AGEG5YR":
        age_map = {
            1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39",
            5: "40–44", 6: "45–49", 7: "50–54", 8: "55–59",
            9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79",
            13: "80+"
        }
        df_clean[feature] = df_clean[feature].map(age_map)

    elif feature == "GenHlth":
        health_map = {
            1: "Excellent",
            2: "Very Good",
            3: "Good",
            4: "Fair",
            5: "Poor"
        }
        df_clean[feature] = df_clean[feature].map(health_map)

    # Group and calculate percentage
    grouped = df_clean.groupby([feature, "Diabetes_binary"]).size().reset_index(name="Count")
    total_per_group = grouped.groupby(feature)["Count"].transform("sum")
    grouped["Percentage"] = (grouped["Count"] / total_per_group) * 100
    grouped["Diabetes Status"] = grouped["Diabetes_binary"].map({0: "No Diabetes", 1: "Diabetes"})

    # Ensure Diabetes is stacked on bottom
    grouped["Diabetes Status"] = pd.Categorical(
        grouped["Diabetes Status"], 
        categories=["Diabetes", "No Diabetes"], 
        ordered=True
    )

    fig = px.bar(
        grouped,
        x=feature,
        y="Percentage",
        color="Diabetes Status",
        color_discrete_map={"Diabetes": "#3a3ebd", "No Diabetes": "#3a79bd"},
        barmode="stack",
        title=f"Diabetes Percentage by {feature}"
    )
    fig.update_layout(xaxis_title=FEATURE_LEGEND[feature], yaxis_title="Percentage")
    return fig

     
 




# ----- FLASK ROUTES -----

@server.route('/')
def index():
    return render_template('index.html')


@server.route("/notebook")
def render_notebook():
    with open("notebook.ipynb") as f:
        nb = nbformat.read(f, as_version=4)
        html_exporter = HTMLExporter()
        body, _ = html_exporter.from_notebook_node(nb)
        return Response(body, mimetype="text/html")

# Define the expected feature names in the correct order
FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

@server.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get JSON data from the request
        print(data)

        # Wrap input in DataFrame using the correct column names
        input_df = pd.DataFrame([data], columns=FEATURE_NAMES)

        # Scale the input data using the pre-loaded scaler
        input_scaled = scaler.transform(input_df)

        # Predict the probability of diabetes (positive class)
        probs = model.predict_proba(input_scaled)
        prob = probs[0][1]  # Probability of positive class (diabetes)

        # Calculate risk percentage
        risk_percent = round(prob * 100, 2)
        print("risk %: ", risk_percent )

        # Determine the risk level based on the percentage
        if risk_percent >= 75:
            risk_level = "High"
        elif risk_percent >= 40:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        # Return the risk percentage and risk level as JSON response
        return jsonify({
            "risk_percent": risk_percent,
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    server.run(debug=True, host="0.0.0.0", port=8000)
