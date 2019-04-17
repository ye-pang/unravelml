import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from layouts import layout_home, layout_eda, layout_eval, layout_unravel
import callbacks

import flask

# default setup

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'), 
    
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='fname', style={'display': 'none'}),
    html.Div(id='data_job', style={'display': 'none'}),
    html.Div(id='train_model', style={'display': 'none'}),
    html.Div(id='model_job', style={'display': 'none'}),
])


# alternative setup

# url_bar_and_content_div = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='page-content'), 
    
#     # Hidden div inside the app that stores the intermediate value
#     html.Div(id='fname', style={'display': 'none'}),
#     html.Div(id='data_job', style={'display': 'none'}),
#     html.Div(id='model_job', style={'display': 'none'})
# ])

# def serve_layout():
#     if flask.has_request_context():
#         return url_bar_and_content_div
#     return html.Div([
#         url_bar_and_content_div,
#         layout_home,
#         layout_eda,
#         layout_eval,
#         layout_unravel
#     ])

# app.layout = serve_layout


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/home':
        return layout_home
    elif pathname == '/data':
        return layout_eda
    elif pathname == '/eval':
        return layout_eval
    elif pathname == '/unravel':
        return layout_unravel
    else:
        return '404'
    
# change these values
PORT = 6006
ADDRESS = '0.0.0.0'

if __name__ == '__main__':
    app.run_server(debug=False, port=PORT, host=ADDRESS, threaded=True)