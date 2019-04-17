# -*- coding: utf-8 -*-

from app import app
import base64
import datetime
import io
import os, sys
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go

from catboost import Pool, CatBoostClassifier
import sklearn.metrics 
from catboost.utils import get_roc_curve
from eli5.permutation_importance import get_score_importances
import shap

from celery.result import AsyncResult
lib_path = '/opt/notebooks/demo/celery/tasks'
sys.path.append(lib_path)
from unravelML import variable_role
from unravelML import train_model


# helper function to write some messages to print
def write_msg_for_df(df):
    num_features = df[df.Role=='feature'].Variable.shape[0]
    num_features_num = df[(df.Role=='feature') & (df.Type=='num')].Variable.shape[0]
    num_features_cat = df[(df.Role=='feature') & (df.Type=='cat')].Variable.shape[0]
    num_features_meta = df[(df.Role=='meta')].Variable.shape[0]
    target_var = df[df.Role=='target'].Variable.values[0]

    t_msg = 'The target variable is **{}**.'.format(target_var)
    f_msg = 'There are **{}** features: \
    **{}** numerical, **{}** categorical.'.format(num_features, num_features_num, num_features_cat)
    m_msg = 'There are **{}** meta features, which will not participate in model training.'.format(num_features_meta)
    
    return t_msg, f_msg, m_msg

# helper function to convert log-odds back to probability
def logit2prob(logit):
    odds = np.exp(logit)
    prob = odds / (1 + odds)
    return prob

def explain_pred(index, scored_test_pool, explainer):
    pred_l1 = ('ID: {}'.format(scored_test_pool.loc[index, :].ID.astype(int)))
    pred_l2 = ('Actual Y: {}'.format(scored_test_pool.loc[index, :].Y.astype(int)))
    pred_l3 = ('Predicted Y: {}'.format(scored_test_pool.loc[index, :].Pred_Y.astype(int)))
    pred_class = 'PredProb_' + str(scored_test_pool.loc[index, :].Pred_Y.astype(int))
    pred_l4 = ('Predicted Probability: {:.2f}'.format(scored_test_pool.loc[index, :][pred_class]))
    pred_l5 = ('Base shap value without any features: {:.2f}'.format(((explainer.expected_value))))
    return [pred_l1, pred_l2, pred_l3, pred_l4, pred_l5]

def explain_row(index, predictions, explainer, shap_values, var_roles, x_test):

    # prediction of the test row
    pred_class = predictions[index]

    # base value of the explainer 
    based_value = explainer.expected_value

    # shap values
    y = shap_values[index,:]

    # sum of higher shap values
    h_shap_sum = sum(y[np.where(y>=0)])

    # sum of lower shap values
    l_shap_sum = sum(y[np.where(y<0)])

    color= np.array(['rgb(255,255,255)']*y.shape[0])
    color[y<0]='rgb(228,81,73)'
    color[y>=0]='rgb(37,144,134)'
    feature_values = np.array([var_roles[var_roles.Feature==each].Variable.values[0] \
                               + ' = ' + str(x_test.iloc[index, :][each]) \
                               for each in x_test.iloc[0, :].index])

    trace1 = go.Bar(
        x=[x_name for _,x_name in sorted(zip(y[np.where(y>=0)], x_test.columns[np.where(y>=0)]), reverse=True)],
        y=sorted(y[np.where(y>=0)], reverse=True),
        text=[txt_name for _,txt_name in sorted(zip(y[np.where(y>=0)], feature_values[np.where(y>=0)]), reverse=True)],
        marker=dict(color=color[np.where(y>=0)].tolist()),
        name= 'Higher', 
        hoverinfo = 'y+text',
        orientation = 'v',
        showlegend = True
    )

    trace2 = go.Bar(
        x=[x_name for _,x_name in sorted(zip(y[np.where(y<0)], x_test.columns[np.where(y<0)]))],
        y=sorted(y[np.where(y<0)]),
        text=[txt_name for _,txt_name in sorted(zip(y[np.where(y<0)], feature_values[np.where(y<0)]))],
        marker=dict(color=color[np.where(y<0)].tolist()),
        name= 'Lower', 
        hoverinfo = 'y+text',
        orientation = 'v',
        showlegend = True
    )

    data = [trace1, trace2]

    layout = go.Layout(
        barmode='stack',
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='Features',
                font=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='SHAP values (Log-Odds)',
                font=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            )
        )
    )

    local_fig = go.Figure(data=data, layout=layout)

    local_fig['layout'].update(
#         title=go.layout.Title(text='Feature Contributions For Predicted Class {}'.format(int(pred_class))),
        title=go.layout.Title(text='Feature Contributions'),
        autosize=True, 
    )

    return local_fig

def return_eval_job(file):
    # do something
    var_roles = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'.pkl')
    cat_mappings = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_catsMap.pkl')
    tuned_params = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_params.pkl')
    y_label_mapping = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_labelsMap.pkl')
    scored_test_pool = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_test_pred.pkl')
    m_test = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_mTest.pkl')  
    x_test = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_xTest.pkl')
    y_test = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_yTest.pkl')

    tuned_model = CatBoostClassifier()
    tuned_model.load_model('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_fitted.dump')

    test_pool = Pool(x_test, np.ravel(y_test), cat_features=cat_mappings.Index.to_list())
    fpr, tpr, thresholds = get_roc_curve(tuned_model, test_pool)
    auc = sklearn.metrics.auc(fpr, tpr)

    lw = 2

    trace1 = go.Scatter(x=fpr, 
                        y=tpr, 
                        text=['Threshold: ' + str(np.round(each, 4)) for each in thresholds],
                        mode='lines', 
                        hoverinfo = 'text+x+y',
                        line=dict(color='darkorange', width=lw),
                        name='ROC curve (area = %0.2f)' % auc
                       )

    trace2 = go.Scatter(x=[0, 1], y=[0, 1], 
                        mode='lines', 
                        line=dict(color='navy', width=lw, dash='dash'),
                        showlegend=False)

    layout = go.Layout(
                        title='Area Under the Receiver Operating Characteristics',
                        xaxis=dict(title='False Positive Rate'),
                        yaxis=dict(title='True Positive Rate', hoverformat = '.2f'), 
                      )

    auc_fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    predictions = tuned_model.predict(test_pool)

    # save classification report to a dictionary
    prfs_report = sklearn.metrics.classification_report(np.ravel(y_test),
                          predictions, 
                          target_names= y_label_mapping.Label.to_list(),
                          output_dict = True
                         )

    # move the nested dict to a dataframe
    prfs_values = []
    for k,v in prfs_report.items():
        for k2, v2 in v.items():
            prfs_values.append([k, k2, v2])

    prfs_df = pd.DataFrame(prfs_values, columns=['Label', 'Metric', 'Value'])

    # set colors 

    N = len(prfs_df.Label.unique()) + \
    sum(1 for i in prfs_df.Label.unique() if i not in ['micro avg', 'macro avg', 'weighted avg']) # account for support accounts

    c = ['hsl('+str(h)+',75%'+',75%)' for h in np.linspace(0, 360, N)]

    trace_groups=[]

    for k in prfs_df.Label.unique():
        _precision = prfs_df[(prfs_df.Label==k) & (prfs_df.Metric=='precision')].Value.values[0]
        _recall = prfs_df[(prfs_df.Label==k) & (prfs_df.Metric=='recall')].Value.values[0]
        _f1 = prfs_df[(prfs_df.Label==k) & (prfs_df.Metric=='f1-score')].Value.values[0]

        trace = go.Bar(
            x=['precision', 'recall', 'f1-score'],
            y=[_precision, _recall, _f1],
            name=k,
            marker=dict(color=c[np.where(prfs_df.Label.unique() == k)[0][0]])
        )
        trace_groups.append(trace) 

    for k in prfs_df.Label.unique():
        if k not in ['micro avg', 'macro avg', 'weighted avg']:
            _support = prfs_df[(prfs_df.Label==k) & (prfs_df.Metric=='support')].Value.values[0]
            trace = go.Bar(
                x=['support'],
                y=[_support],
                yaxis='y2',
                name=k,
                marker=dict(color=c[np.where(prfs_df.Label.unique() == k)[0][0]]), 
                showlegend = False
            )
            trace_groups.append(trace) 

    data = trace_groups
    layout = go.Layout(
        title=go.layout.Title(
            text='Classification Performance Report',
            xref='paper',
        ),
        legend=dict(orientation="h"),
        barmode='group', 
        yaxis=dict(
            title='Percentage',
            range=[0, 1]
        ),
        yaxis2=dict(
            title='Count',
            range=[0, 10000],
            overlaying='y',
            side='right'
        ),
        #paper_bgcolor = 'rgb(233,233,233)', 
        #plot_bgcolor = 'rgb(233,233,233)'
    )

    crpt_fig = go.Figure(data=data, layout=layout)
    
        
    # return something
    return [
            #html.H3('Fitted model loaded...'), 
            dcc.Markdown('''
**Auto-tuned hyperparameters** are the following:
* max_depth = **{}**
* l2_leaf_reg = **{}**
* learning_rate = **{:.3f}**
* iterations = **{}**
            '''.format(tuned_params.loc['max_depth'].Value,
                       tuned_params.loc['l2_leaf_reg'].Value,
                       tuned_params.loc['learning_rate'].Value,
                       tuned_params.loc['iterations'].Value)), 
            html.Div(dcc.Markdown(children = '''
**AUC** (Area Under The Curve) - **ROC** (Receiver Operating Characteristics) curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between target classes.
            '''), style={'width': '80vh'}),
            dcc.Graph(
                id='auc-graph',
                figure = auc_fig, 
                style={'width': '90vh'}
            ),
            html.Div(dcc.Markdown('''
**Classification Report** below provides the scores (precision, recall, f1) corresponding to every class as well as different weighing schemas using the support counts of the target class. We aim to explain the accuracy of the classifier in classifying the data points in that particular class compared to all other classes.
            '''), style={'width': '80vh'}),
            dcc.Graph(
                id='crpt-graph',
                figure = crpt_fig, 
                style={'width': '90vh'}
            ),
    ]

def return_unravel(file):
    # do something
    var_roles = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'.pkl')
    cat_mappings = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_catsMap.pkl')
    tuned_params = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_params.pkl')
    y_label_mapping = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_labelsMap.pkl')
    scored_test_pool = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_test_pred.pkl')
    m_test = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_mTest.pkl')  
    x_test = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_xTest.pkl')
    y_test = pd.read_pickle('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_yTest.pkl')

    tuned_model = CatBoostClassifier()
    tuned_model.load_model('/opt/notebooks/demo/celery/processed/'+file.split('.')[0]+'_fitted.dump')

    test_pool = Pool(x_test, np.ravel(y_test), cat_features=cat_mappings.Index.to_list())
    fpr, tpr, thresholds = get_roc_curve(tuned_model, test_pool)
    auc = sklearn.metrics.auc(fpr, tpr)
    predictions = tuned_model.predict(test_pool)

    # ... load data, define score function
    def score(X, y):
        y_pred = tuned_model.predict(X)
        return sklearn.metrics.roc_auc_score(y, y_pred)

    base_score, score_decreases = get_score_importances(score, x_test.to_numpy(), y_test.to_numpy(), 
                                                       random_state=42, 
                                                       n_iter=15)
    feature_importances = np.mean(score_decreases, axis=0)
    
    N = len(x_test.columns) # Number of boxes

    c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

    data = [{
        'y': np.array(score_decreases)[:, i]*100, 
        'name': var_roles[var_roles.Feature==x_test.columns[i]].Variable.values[0],
        'type':'box',
        'marker':{'color': c[i]}
        } for i in range(int(N))]

    # format the layout
    layout = go.Layout(
                title=go.layout.Title(
                    text='Mean Decrease In AUC',
                    xref='paper',
                ),
                legend=dict(orientation="h"),
                barmode='group', 
                xaxis=dict(
                    showgrid=False,
                    zeroline=False, 
                    tickangle=60,
                    showticklabels=False
                ),
                yaxis=dict(
                    title='Basis Points Drop',
                    zeroline=True,
                    gridcolor='white', 
                    #range=[0, 10]
                ), 
                #paper_bgcolor = 'rgb(233,233,233)', 
                #plot_bgcolor = 'rgb(233,233,233)'
    )
    
    auc_features_fig = go.Figure(data=data, layout=layout)
    
    joint_importance = tuned_model.get_feature_importance(test_pool,prettified = True, type = "Interaction")

    features1 = [item[0] for item in joint_importance]
    features2 = [item[1] for item in joint_importance]
    joint_imp = [item[2] for item in joint_importance]
    str_features = []
    first_feature = []
    second_feature = []

    for i in range(0,len(features1)):
        for j in range(0,len(x_test.columns)):
            if features1[i] == j:
                str_a = var_roles[var_roles.Feature==x_test.columns[j]].Variable.values[0]   #x_test.columns[j]
            if features2[i] == j:
                str_b = var_roles[var_roles.Feature==x_test.columns[j]].Variable.values[0]   #x_test.columns[j]
        str_features.append(str_a + " & " + str_b)
        first_feature.append(str_a)
        second_feature.append(str_b)

    explainer = shap.TreeExplainer(tuned_model, model_output='margin')
    shap_values = explainer.shap_values(test_pool)    
    
    # scatter plot (this is not dynamic yet...)
    x = x_test.iloc[:,features1[0]]
    y = pd.Series(shap_values[:,features1[0]])
    #x = x.apply(lambda n: n+0.1*(np.random.random_sample()-0.5))
    tracefull1 = go.Scatter(
        x = x,
        y = y,
        name = str_features[0],
        mode = 'markers',
        marker=dict(
                size=6,
                cmax=len(x_test.iloc[:,features2[0]].cat.categories),
                #cmin=0,
                color=[each for each in x_test.iloc[:,features2[0]]],
                #color=[x_test.iloc[:,features2[0]].cat.categories.to_list().index(each) for each in x_test.iloc[:,features2[0]]],
                colorbar=dict(
                    #title='Colorbar', 
                    tick0=0,
                    dtick=1
                ),
                colorscale='RdBu'
            ),
    )

    data = [tracefull1]

    layout = go.Layout(
        barmode='stack',
        title=go.layout.Title(
            text='Feature Importance of ' + str_features[0],
            xref='paper',
            x=0.5
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text=first_feature[0],
                font=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='SHAP values',
                font=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            )
        ),
        annotations=[
            dict(
                x=1.05,
                y=1.05,
                align="right",
                valign="top",
                text=second_feature[0],
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="top"
            )
        ]
    )

    interactions_fig = go.Figure(data=data, layout=layout)
    
    return [
        html.Div(dcc.Markdown(children = '''
One good way to determine global feature importance is through a method called **mean decrease in a metric** by dropping a feature randomly and run multiple permutations on the test data. Here, we use **AUC** as the metric and ran **15** permutations on **{}** rows of the test dataset:
            '''.format(y_test.shape[0])), 
                 #style={'width': '80vh'}
                ), 
        
         dcc.Graph(
            id='permutation-graph',
            figure = auc_features_fig
         ),
        html.Br(),
        html.Div(dcc.Markdown(children = '''
A state of art technique to determine feature importance at the row prediction level uses attributions from **cooperative game theory**. Here's a visualization of feature attributions using log-odds as the 
**Shapley** values for the **1st row** in the test dataset:
            '''), 
                 #style={'width': '80vh'}
                ), 
        
        html.Div(children=[
            dcc.Markdown('{}'.format(explain_pred(0, scored_test_pool, explainer)[0])),
            dcc.Markdown('{}'.format(explain_pred(0, scored_test_pool, explainer)[1])),
            dcc.Markdown('{}'.format(explain_pred(0, scored_test_pool, explainer)[2])),
            dcc.Markdown('{}'.format(explain_pred(0, scored_test_pool, explainer)[3])),
            dcc.Markdown('{}'.format(explain_pred(0, scored_test_pool, explainer)[4])),
            #dcc.Markdown('Line2'),
        ], style={'width': '80vh'}
                ),
        
        dcc.Graph(
            id='shap-graph',
            figure = explain_row(0, predictions, explainer, shap_values, var_roles, x_test), 
            #style={'width': '90vh'}
        ), 
        
        html.Div(
            dcc.Markdown('''
Variable interactions are fun, and here are top **3** most important pairwise interactions:
* {}
* {}
* {}
            '''.format(str_features[0], str_features[1], str_features[2])
            ), 
            
        ), 
        
        html.Br(),
        
        dcc.Markdown("Here's a scatter plot for the top interaction. For example, here, we can theorize that most people with good first payment history have lower chance of credit card default, but younger people with good first payment history seem to have even lower chance of credit card default."),
        
        dcc.Graph(
            id='interactions-graph',
            figure = interactions_fig, 
            #style={'width': '90vh'}
        ), 
        
        dcc.Markdown('''
**We hope you enjoyed our tool! Please feel free to contact us with any comments or questions.**
        '''),
        
    ]
                       

#######################################
### Upload Data Callbacks             
#######################################   

# check file uploaded is on the server already

@app.callback([Output('uploaded-result', 'children'),
               Output('to_data', 'style')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])

def verify_file(contents, filename):
    if contents is None:
        raise PreventUpdate
    elif filename.split('.')[1] != 'xlsx':
        return 'Please upload a `.xlsx` file!'
    elif os.path.isfile('/opt/notebooks/demo/uploads/' + filename):
        return '`{}` has already been uploaded, skipping file upload...'.format(filename), {'display': 'block'}
    else:    
        data = contents.encode("utf8").split(b";base64,")[1]
        with open(os.path.join('/opt/notebooks/demo/uploads/', filename), "wb") as fp:
            fp.write(base64.decodebytes(data))
        return 'File uploaded: `{}`'.format(filename), {'display': 'block'}


# store file name in hidden div to pass it around    
@app.callback(Output('fname', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])  

def store_name(contents, filename):
    if contents is None:
        raise PreventUpdate
    else:    
        return filename

#######################################
### Data View Callbacks             
#######################################   

# load file name from hidden div    
# @app.callback(
#     [Output('eda-fname', 'children')],
#     [Input('fname', 'children')])
# def display_value(children):
#     if children is None:
#         raise PreventUpdate
#     elif children.split('.')[1] != 'xlsx':
#         raise PreventUpdate
#     else:
#         return 'File uploaded: `{}`'.format(children)
    

# create job for load eda table     
@app.callback(
    Output('data_job', 'children'),
    [Input('fname', 'children')])

def add_content(children):
    if children is None:
        raise PreventUpdate
    elif children.split('.')[1] != 'xlsx':
        raise PreventUpdate
    elif os.path.isfile('/opt/notebooks/demo/celery/processed/' + children.split('.')[0] + '.pkl'):
        raise PreventUpdate
    else:
        # trigger task to process the spreadsheet
        r = variable_role.delay(children)
        return r.task_id
    

# check eda-loading job and output table
@app.callback(
    Output('eda-content', 'children'),
    [Input('eda-loading', 'n_intervals'), 
     Input('data_job', 'children'), 
     Input('fname', 'children')]) 

def check_state_of_task(n, job, file):
    if file is None:
        raise PreventUpdate
    elif file.split('.')[1] != 'xlsx':
        raise PreventUpdate
    elif job is None and os.path.isfile('/opt/notebooks/demo/celery/processed/' + file.split('.')[0] + '.pkl'):
        df = pd.read_pickle('/opt/notebooks/demo/celery/processed/' + file.split('.')[0] + '.pkl')
        t_msg, f_msg, m_msg = write_msg_for_df(df)
  
        return [dcc.Markdown(t_msg),
                dcc.Markdown(f_msg), 
                dcc.Markdown(m_msg),
                html.Br(), 
                dash_table.DataTable(
                    id='var_roles_table',
                    columns=[{"name": i, "id": i} for i in df[df.columns.difference(['count'])].columns],
                    data=df[df.columns.difference(['count'])].to_dict("rows"), 
                    style_table={
                        'minWidth': '0px', 'maxWidth': '500px',
                        'whiteSpace': 'normal',
                        'maxHeight': '300',
                        'overflowY': 'scroll'
                    },
                )]
    elif job is None:
        raise PreventUpdate
    elif job is not None and variable_role.AsyncResult(job).state=='SUCCESS':
        df = pd.read_pickle('/opt/notebooks/demo/celery/processed/' + file.split('.')[0] + '.pkl')
        t_msg, f_msg, m_msg = write_msg_for_df(df)

        return [#dcc.Markdown('Job `{}` finishied processing'.format(job) + '<br />Here is the result:'),
                dcc.Markdown(t_msg),
                dcc.Markdown(f_msg), 
                dcc.Markdown(m_msg),
                html.Br(), 
                dash_table.DataTable(
                    id='var_roles_table',
                    columns=[{"name": i, "id": i} for i in df[df.columns.difference(['count'])].columns],
                    data=df[df.columns.difference(['count'])].to_dict("rows"), 
                    style_table={
                        'minWidth': '0px', 'maxWidth': '500px',
                        'whiteSpace': 'normal',
                        'maxHeight': '300',
                        'overflowY': 'scroll'
                    },
                )]
    elif job is not None:
        return dcc.Markdown('`{}` processing'.format(job) + ', please wait...')             
    else:
        raise PreventUpdate

        
# disable eda-loading loop and show next link 
@app.callback(
    [Output('eda-loading', 'disabled'),
     Output('to_eval', 'style'),
     Output('train_model', 'children')],
    [Input('var_roles_table', 'data')])   

def disable_eda_loading(data):
    if data is not None:
        return True, {'display': 'block'}, 'Yes'
    else:
        raise PreventUpdate

#######################################
### Model Evaluation Callbacks             
#######################################                   

# create job for training a model  
@app.callback(
    Output('model_job', 'children'),
    [Input('train_model', 'children'),
     Input('fname', 'children')
    ])

def train_model_job(to_train,file_name):
    if to_train is None:
        raise PreventUpdate
    elif file_name.split('.')[1] != 'xlsx':
        raise PreventUpdate
    elif os.path.isfile('/opt/notebooks/demo/celery/processed/' + file_name.split('.')[0] + '_fitted.dump'):
        raise PreventUpdate
    elif to_train == 'Yes':
        # trigger task for model training
        r = train_model.delay(file_name)
        return r.task_id
    else:
        PreventUpdate

# check eval-loading job and populate content div
@app.callback(
    Output('eval-content', 'children'),
    [Input('eval-loading', 'n_intervals'), 
     Input('model_job', 'children'), 
     Input('fname', 'children')
    ])         

def check_model_eval(n, job, file):
    if file is None:
        return ['Please upload a dataset first.', 
                html.Br(), 
                dcc.Link('Back to home', href='/home')]
    elif job is None \
    and os.path.isfile('/opt/notebooks/demo/celery/processed/' + \
                       file.split('.')[0] + '_fitted.dump'):
        return return_eval_job(file)
    elif job is not None and train_model.AsyncResult(job).state=='SUCCESS':
        return return_eval_job(file)
    elif job is not None:
        return dcc.Markdown('`{}` processing'.format(job) + ', please wait...')             
    else:
        raise PreventUpdate

# disable eval-loading loop and show next link 
@app.callback(
    [Output('eval-loading', 'disabled'),
     Output('to_unravel', 'style')],
    [Input('auc-graph', 'figure')])   

def disable_eda_loading(figure):
    if figure is not None:
        return True, {'display': 'block'}
    else:
        raise PreventUpdate        
        
#######################################
### Interpretability Callbacks             
#######################################    

# populate unravel content div
@app.callback(
    Output('unravel-content', 'children'),
    [Input('fname', 'children')]
) 

def check_unravel(file):
    if file is None:
        return ['Please upload a dataset first.', 
                html.Br(), 
                dcc.Link('Back to home', href='/home')]
    elif os.path.isfile('/opt/notebooks/demo/celery/processed/' + \
                       file.split('.')[0] + '_fitted.dump'):
        return return_unravel(file)        
    else:
        raise PreventUpdate

# # disable unravel-loading loop 
# @app.callback(
#     Output('unravel-loading', 'disabled'),
#     Input('permutation-graph', 'figure'))   

# def disable_unravel_loading(figure):
#     if figure is not None:
#         return True
#     else:
#         raise PreventUpdate     