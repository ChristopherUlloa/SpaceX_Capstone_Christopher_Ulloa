
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

df = pd.read_csv('data/sample_launches.csv')

app = Dash(__name__)
app.title = "SpaceX Launches"

app.layout = html.Div([
    html.H2("SpaceX Launches — Demo Dashboard"),
    html.Div([
        html.Div([
            html.Label("Launch Site"),
            dcc.Dropdown(
                id='site',
                options=[{'label': s, 'value': s} for s in sorted(df['LaunchSite'].unique())],
                multi=True
            )
        ], style={'width':'30%','display':'inline-block'}),
        html.Div([
            html.Label("Orbit"),
            dcc.Dropdown(
                id='orbit',
                options=[{'label': s, 'value': s} for s in sorted(df['Orbit'].unique())],
                multi=True
            )
        ], style={'width':'30%','display':'inline-block','marginLeft':'20px'}),
    ]),
    dcc.Graph(id='scatter'),
    dcc.Graph(id='bar')
])

@app.callback(
    Output('scatter','figure'),
    Output('bar','figure'),
    Input('site','value'),
    Input('orbit','value')
)
def update(site, orbit):
    dff = df.copy()
    if site: dff = dff[dff['LaunchSite'].isin(site)]
    if orbit: dff = dff[dff['Orbit'].isin(orbit)]
    fig_scatter = px.scatter(dff, x='FlightNumber', y='PayloadMass', color='Class',
                             hover_data=['LaunchSite','Orbit','Serial'])
    fig_bar = px.bar(dff.groupby('LaunchSite')['Class'].mean().reset_index(),
                     x='LaunchSite', y='Class', title='Success Rate by Site')
    fig_bar.update_yaxes(range=[0,1])
    return fig_scatter, fig_bar

if __name__ == "__main__":
    app.run_server(debug=True)
