import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd

app = dash.Dash(__name__)

# Función para el header
def header():
    return html.Div(
        style={"display": "flex", "alignItems": "center", "padding": "10px"},
        children=[
            html.Img(src="/assets/SICbanner.png", style={"height": "70px", "padding-right": "20px"}),
            html.H1("Predicción de demoras en programa de resolución de quejas",
                    style={"fontSize": "40px", "textAlign": "center", "color":'white'})
        ]
    )

# Función para el panel izquierdo de selección
def left_panel():
    return html.Div(
        style={"backgroundColor": "#2b2b2b", "padding": "10px", "width": "20%", "margin": "10px"},
        children=[
            html.P("Seleccione las características de su queja", 
                    style={"fontSize": "15px", 'color':'white', "padding": "10px"}),
            dcc.Dropdown(
                id="caracteristica1",
                options=[{"label": f"Característica {i}", "value": i} for i in range(1, 9)],
                placeholder="Característica 1",
                style={"marginBottom": "10px"}
            ),
            dcc.Dropdown(
                id="caracteristica2",
                options=[{"label": f"Característica {i}", "value": i} for i in range(1, 9)],
                placeholder="Característica 2",
                style={"marginBottom": "10px"}
            ),
            dcc.Dropdown(
                id="caracteristica3",
                options=[{"label": f"Característica {i}", "value": i} for i in range(1, 9)],
                placeholder="Característica 3",
                style={"marginBottom": "10px"}
            ),
            # Agrega más dropdowns según lo necesites
        ]
    )

# Función para los indicadores principales
def main_indicators():
    return html.Div(
        style={"display": "flex", "justifyContent": "space-around"},
        children=[
            html.Div(
                style={"width": "20%", "backgroundColor": "#333333", "padding": "10px", "borderRadius": "5px"},
                children=[
                    html.H4("Variable principal 1", style={"color": "white", "fontSize": "20px"}),
                    html.P("Last value:", style={"color": "#888", "textAlign": "center"}),
                    html.H4("$ 500.000", style={"fontSize": "40px", "textAlign": "center", "marginTop": "-15px"}),
                    html.P("COP", style={"color": "#888", "textAlign": "center", "marginTop": "-15px"}),
                ]
            ),
            html.Div(
                style={"width": "20%", "backgroundColor": "#333333", "padding": "10px", "borderRadius": "5px"},
                children=[
                    html.H4("Variable principal 2" , style={"color": "white", "fontSize": "20px"}),
                    html.P("Last value:", style={"color": "#888", "textAlign": "center"}),
                    html.H4("E 2", style={"fontSize": "40px", "textAlign": "center", "marginTop": "-15px"} ),
                    html.P("Estrato", style={"color": "#888", "textAlign": "center", "marginTop": "-15px"}),
                ]
            ),
            html.Div(
                style={"width": "20%", "backgroundColor": "#333333", "padding": "10px", "borderRadius": "5px"},
                children=[
                    html.H4("Variable principal 3" , style={"color": "white", "fontSize": "20px"}),
                    html.P("Last value:", style={"color": "#888", "textAlign": "center"}),
                    html.H4("SEDE CUN", style={"fontSize": "40px", "textAlign": "center", "marginTop": "-15px"} ),
                    html.P("Lugar de origen", style={"color": "#888", "textAlign": "center", "marginTop": "-15px"}),
                ]
            ),
            html.Div(
                style={"width": "20%", "backgroundColor": "#333333", "padding": "10px", "borderRadius": "5px"},
                children=[
                    html.H4("Variable principal 4" , style={"color": "white", "fontSize": "20px"}),
                    html.P("Last value:", style={"color": "#888", "textAlign": "center"}),
                    html.H4("05/03/2022", style={"fontSize": "40px", "textAlign": "center", "marginTop": "-15px"} ),
                    html.P("DD/MM/AAAA", style={"color": "#888", "textAlign": "center", "marginTop": "-15px"}),
                ]
            ),
            # Agrega más indicadores según lo necesites
        ]
    )

# Función para la sección de gráficos
def graphs_section():
    return html.Div(
        style={"display": "flex", "justifyContent": "space-around", "marginTop": "20px"},
        children=[
            dcc.Graph(
                id="bar_graph",
                style={"backgroundColor": "#2b2b2b", "borderRadius": "5px", "padding": "0", "margin": "0"},
            ),
            html.Div(
                style={"backgroundColor": "#333333", "padding": "10px", "borderRadius": "5px"},
                children=[
                    html.H4("Predicción:", style={"color": "white", "fontSize": "20px"}),
                    html.H2("100", style={"fontSize": "100px", "textAlign": "center"}),
                    html.P("días", style={"color": "#888", "textAlign": "center"}),
                ]
            ),
        ]
    )

# Definir el layout de la aplicación
app.layout = html.Div(
    style={"backgroundColor": "#1e1e1e", "color": "#2cfec1", "fontFamily": "Arial"},
    children=[
        header(),
        html.Div(
            style={"display": "flex"},
            children=[
                left_panel(),
                html.Div(
                    style={"width": "75%", "padding": "10px", "margin": "10px"},
                    children=[
                        main_indicators(),
                        graphs_section()
                    ]
                )
            ]
        )
    ]
)

# Callback para actualizar el gráfico de barras
@app.callback(
    Output("bar_graph", "figure"),
    [Input("caracteristica1", "value"), Input("caracteristica2", "value")]
)
def update_bar_graph(caracteristica1, caracteristica2):
    # Genera datos de ejemplo, puedes reemplazar con datos reales
    y_values = np.random.randint(1, 10, size=5)
    fig = go.Figure(go.Bar(
        y=["Característica 1", "Característica 2", "Característica 3", "Característica 4", "Característica 5"],
        x=y_values,
        orientation='h'
    ))


    fig.update_layout(
        height=400,  # Ajusta la altura del gráfico en píxeles
        width=700,   # Ajusta el ancho del gráfico en píxeles
        paper_bgcolor='#2b2b2b',
        plot_bgcolor='#2b2b2b',
        font_color='#2cfec1',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    return fig

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run_server(debug=True)