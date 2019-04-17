import dash_core_components as dcc
import dash_html_components as html

# layout for home page
layout_home = html.Div([
    
    html.H3(children='Demo: Supervised Classification'),
    
    html.Br(),
    dcc.Markdown(
    """
Please upload a `.xlsx` file with the following formatting:
* Row 1: Variable Name 
* Row 2: Variable Role (accepted values - 'meta', 'feature', 'target')
* Row 3: Variable Type (accepted values - 'num' or 'cat')
* Row 4 ... Row N : data
* __Tip__: Dates and text can be broken into separate columns 
    """),
 
    html.Br(),
    
    dcc.Upload(
        id='upload-data', 
        children=html.Button(
            children='Upload File'
        ),
        multiple=False
    ),
    
    html.Br(),

    dcc.Markdown(id='uploaded-result', children='No file uploaded yet...'),
       
    html.Br(),

    dcc.Link(id='to_data', children='Next', href='/data', refresh=False, style={'display': 'none'})
])

# layout for data view page
layout_eda = html.Div([
#     dcc.Markdown(id='eda-fname'),
#     html.Br(),
    dcc.Interval(
        id='eda-loading',
        interval=10*1000, # in milliseconds
        n_intervals=0
    ),
    html.Div(id='eda-content'), 
    html.Br(),
    dcc.Link(id='to_eval', children='Next', href='/eval', refresh=False, style={'display': 'none'})
])

# layout for model eval page 
layout_eval = html.Div([
    dcc.Interval(
        id='eval-loading',
        interval=10*1000, # in milliseconds
        n_intervals=0, 
    ),
    html.Div(id='eval-content'), 
    html.Br(),
    dcc.Link(id='to_unravel', children='Next', href='/unravel', refresh=False, style={'display': 'none'})
])

# layout for interpretability page
layout_unravel = html.Div([
#     dcc.Interval(
#         id='unravel-loading',
#         interval=10*1000, # in milliseconds
#         n_intervals=0, 
#     ),
    html.Div(id='unravel-content'), 
])