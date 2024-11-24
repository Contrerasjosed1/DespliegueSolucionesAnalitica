import dash
import joblib
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import requests

pipeline = joblib.load('modelo_prediccion_pipeline.pkl')
API_URL = "http://localhost:8001/api/v1/predict"  # Cambia localhost por la IP de tu servidor si no está local

# Cargar datos
df = pd.read_csv('SegundaEntrega/ARREGLO_DIRECTO.csv', delimiter=';')

# Procesamiento de datos
REGION_DEPTO = {
    'ATLANTICO': 'Caribe',
    'BOLIVAR': 'Caribe',
    'CESAR': 'Caribe',
    'CORDOBA': 'Caribe',
    'SUCRE': 'Caribe',
    'SAN ANDRES': 'Caribe',
    'CAUCA': 'Pacífica',
    'VALLE DEL CAUCA': 'Pacífica',
    'NARIÑO': 'Pacífica',
    'BOGOTA': 'Andina',
    'CUNDINAMARCA': 'Andina',
    'HUILA': 'Andina',
    'TOLIMA': 'Andina',
    'QUINDIO': 'Andina',
    'RISARALDA': 'Andina',
    'SANTANDER': 'Andina',
    'N. DE SANTANDER': 'Andina',
    'META': 'Orinoquia'}

# Limpieza de datos
df['VALOR_PRODUCTO'] = df['VALOR_PRODUCTO'].replace({r'[^\d.]': '', 'INDETERMINADO': None}, regex=True)
df['VALOR_PRODUCTO'] = pd.to_numeric(df['VALOR_PRODUCTO'], errors='coerce')
promedio = df.loc[df['VALOR_PRODUCTO'] > 0, 'VALOR_PRODUCTO'].mean()
df['VALOR_PRODUCTO'] = df['VALOR_PRODUCTO'].apply(lambda x: promedio if pd.isna(x) or x == 0 else x)
df['REGION'] = df['UNIDAD_DEPARTAMENTO'].map(REGION_DEPTO)

# Selección de variables
variables_interes = ['REGION', 'ATENCION_TEMA', 'PERSONA_RANGO_EDAD', 'PERSONA_GENERO', 'PERSONA_PROFESION', 'TIPO_PRODUCTO', 'VALOR_PRODUCTO', 'DURACION']
data = df[variables_interes].copy()

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
            html.P("Región:", style={"color": "#888"}),
            dcc.Dropdown(
                id="REGION",
                options=[{"label": region, "value": region} for region in data['REGION'].unique()],
                placeholder="Región",
                style={"marginBottom": "10px", "color": "white", "backgroundColor": "#333333"}
            ),
            html.P("Tema de atención:", style={"color": "#888"}),
            dcc.Dropdown(
                id="ATENCION_TEMA",
                options=[{"label": ATENCION_TEMA, "value": ATENCION_TEMA} for ATENCION_TEMA in data['ATENCION_TEMA'].unique()],
                placeholder="Tema de atención",
                style={"marginBottom": "10px", "color": "white", "backgroundColor": "#333333"}
            ),
            html.P("Rango de edad de la persona:", style={"color": "#888"}),
            dcc.Dropdown(
                id="PERSONA_RANGO_EDAD",
                options=[{"label": PERSONA_RANGO_EDAD, "value": PERSONA_RANGO_EDAD} for PERSONA_RANGO_EDAD in data['PERSONA_RANGO_EDAD'].unique()],
                placeholder="Persona rango edad",
                style={"marginBottom": "10px", "color": "white", "backgroundColor": "#333333"}
            ),
            html.P("Género:", style={"color": "#888"}),
            dcc.Dropdown(
                id="PERSONA_GENERO",
                options=[{"label": PERSONA_GENERO, "value": PERSONA_GENERO} for PERSONA_GENERO in data['PERSONA_GENERO'].unique()],
                placeholder="Género",
                style={"marginBottom": "10px", "color": "white", "backgroundColor": "#333333"}
            ),
            html.P("Profesión:", style={"color": "#888"}),
            dcc.Dropdown(
                id="PERSONA_PROFESION",
                options=[{"label": PERSONA_PROFESION, "value": PERSONA_PROFESION} for PERSONA_PROFESION in data['PERSONA_PROFESION'].unique()],
                placeholder="Profesión",
                style={"marginBottom": "10px", "color": "white", "backgroundColor": "#333333"}
            ),
            html.P("Tipo de producto:", style={"color": "#888"}),
            dcc.Dropdown(
                id="TIPO_PRODUCTO",
                options=[{"label": TIPO_PRODUCTO, "value": TIPO_PRODUCTO} for TIPO_PRODUCTO in data['TIPO_PRODUCTO'].unique()],
                placeholder="Tipo de producto",
                style={"marginBottom": "10px", "color": "white", "backgroundColor": "#333333"}
            ),
            html.P("Valor del producto:", style={"color": "#888"}),
            dcc.Dropdown(
                id="VALOR_PRODUCTO",
                options=[{"label": VALOR_PRODUCTO, "value": VALOR_PRODUCTO} for VALOR_PRODUCTO in data['VALOR_PRODUCTO'].unique()],
                placeholder="Valor del producto",
                style={"marginBottom": "10px", "color": "white", "backgroundColor": "#333333"}
            ),
            html.Button("Calcular", id="calculate_button", n_clicks=0, style={"marginTop": "20px", "width": "100%"})

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
                    html.H4("Valor del producto", style={"color": "white", "fontSize": "20px"}),
                    html.P("Last value:", style={"color": "#888", "textAlign": "center"}),
                    html.H4(id= 'ValorProducto_output', children="$ 500.000", style={"fontSize": "40px", "textAlign": "center", "marginTop": "-15px"}),
                    html.P("COP", style={"color": "#888", "textAlign": "center", "marginTop": "-15px"}),
                ]
            ),
            html.Div(
                style={"width": "20%", "backgroundColor": "#333333", "padding": "10px", "borderRadius": "5px"},
                children=[
                    html.H4("Región" , style={"color": "white", "fontSize": "20px"}),
                    html.P("Last value:", style={"color": "#888", "textAlign": "center"}),
                    html.H4(id='Region_output', children="E 2", style={"fontSize": "40px", "textAlign": "center", "marginTop": "-15px"} ),
                    # html.P("Estrato", style={"color": "#888", "textAlign": "center", "marginTop": "-15px"}),
                ]
            ),
            html.Div(
                style={"width": "20%", "backgroundColor": "#333333", "padding": "10px", "borderRadius": "5px"},
                children=[
                    html.H4("Genero" , style={"color": "white", "fontSize": "20px"}),
                    html.P("Last value:", style={"color": "#888", "textAlign": "center"}),
                    html.H4(id='Genero_output', children="SEDE CUN", style={"fontSize": "40px", "textAlign": "center", "marginTop": "-15px"} ),
                    # html.P("Lugar de origen", style={"color": "#888", "textAlign": "center", "marginTop": "-15px"}),
                ]
            ),
            html.Div(
                style={"width": "20%", "backgroundColor": "#333333", "padding": "10px", "borderRadius": "5px"},
                children=[
                    html.H4("Profesion" , style={"color": "white", "fontSize": "20px"}),
                    html.P("Last value:", style={"color": "#888", "textAlign": "center"}),
                    html.H4(id='Profesion_output', children="05/03/2022", style={"fontSize": "40px", "textAlign": "center", "marginTop": "-15px"} ),
                    # html.P("DD/MM/AAAA", style={"color": "#888", "textAlign": "center", "marginTop": "-15px"}),
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
                    # html.H2("100", style={"fontSize": "100px", "textAlign": "center"}),
                    html.H2(id="prediction_output", children="100", style={"fontSize": "100px", "textAlign": "center"}),
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
    Output("prediction_output", "children"),
    Output("ValorProducto_output", "children"),
    Output("Region_output", "children"),
    Output("Genero_output", "children"),
    Output("Profesion_output", "children"),
    [Input('calculate_button', 'n_clicks')],
    [State("REGION", "value"), State("ATENCION_TEMA", "value"), State("PERSONA_RANGO_EDAD", "value"), 
     State("PERSONA_GENERO", "value"), State("PERSONA_PROFESION", "value"), State("TIPO_PRODUCTO", "value"),
     State("VALOR_PRODUCTO", "value")]
)
def update_bar_graph(n_clicks, REGION, ATENCION_TEMA, PERSONA_RANGO_EDAD, PERSONA_GENERO, PERSONA_PROFESION, TIPO_PRODUCTO, VALOR_PRODUCTO):
    if n_clicks > 0:
        # Construir el payload para la API
        input_data = {
            "REGION": REGION,
            "ATENCION_TEMA": ATENCION_TEMA,
            "PERSONA_RANGO_EDAD": PERSONA_RANGO_EDAD,
            "PERSONA_GENERO": PERSONA_GENERO,
            "PERSONA_PROFESION": PERSONA_PROFESION,
            "TIPO_PRODUCTO": TIPO_PRODUCTO,
            "VALOR_PRODUCTO": VALOR_PRODUCTO
        }
        
        try:
            # Enviar solicitud POST a la API
            response = requests.post(API_URL, json=input_data)
            
            # Verificar el código de estado de la respuesta
            if response.status_code == 200:
                prediction = response.json().get("prediction", ["Error"])[0]

                # Crear el histograma para `VALOR_PRODUCTO`
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df['VALOR_PRODUCTO'],
                    nbinsx=3000,
                    marker=dict(color='#2cfec1'),
                    name="Distribución de VALOR_PRODUCTO"
                ))
                fig.add_trace(go.Scatter(
                    x=[VALOR_PRODUCTO, VALOR_PRODUCTO],
                    y=[0, max(np.histogram(df['VALOR_PRODUCTO'], bins=3000)[0]/2)],
                    mode="lines",
                    line=dict(color="red", width=3),
                    name="Valor Seleccionado"
                ))
                fig.update_layout(
                    title="Histograma de Valor del Producto",
                    xaxis_title="Valor del Producto",
                    yaxis_title="Frecuencia",
                    paper_bgcolor='#2b2b2b',
                    plot_bgcolor='#2b2b2b',
                    font_color='#2cfec1',
                    height=400,
                    width=700,
                    xaxis_range=[0, 100000000 if VALOR_PRODUCTO < 100000000 else VALOR_PRODUCTO]
                )

                return fig, f"{prediction:.2f}", f"$ {VALOR_PRODUCTO:.1f}", REGION, PERSONA_GENERO, PERSONA_PROFESION
            else:
                return go.Figure(), "Error en la API", '-', '-', '-', '-'

        except Exception as e:
            return go.Figure(), f"Error: {str(e)}", '-', '-', '-', '-'

    # Valores por defecto si el botón no se clicó
    return go.Figure(), "--", '-', '-', '-', '-'

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run_server(debug=True)