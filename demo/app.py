# -*- coding: utf-8 -*-

import dash

app = dash.Dash(__name__, static_folder='assets')

app.css.append_css({'external_url': 'assets/bWLwgP.css'})
app.css.append_css({'external_url': 'assets/body.css'})

app.config['suppress_callback_exceptions'] = True