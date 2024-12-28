#%% 

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Sample data for the five subgroups
categories = ["1","2","3","4","5"]
x_values = np.arange(len(categories))

# Means and standard deviations for two sets
means_set1 = [42.9, 13.0, 72.6, 70.13, 84.8]
std_set1 = [4.6, 2.2, 0.4, 6.4, 1.4]
stde_set1 = [x/np.sqrt(4) for x in std_set1]

# means_set2 = [54.6, 19.1, 76.7, 90.5, 95.4]
# std_set2 = [0.5, 2.0, 0.9, 0.5, 1.1]
# stde_set2 = [x/np.sqrt(3) for x in std_set2]


means_set2 = [57.0, 19.2, 78.0, 93.8, 93.5]
std_set2 = [0.61, 1.6, 0.9, 0.24, 0.97]
stde_set2 = [x/np.sqrt(3) for x in std_set2]

# Create a subplot layout with 5 vertical subplots
fig = go.Figure()

# Adding bars for the first set
fig.add_trace(go.Bar(
    x=categories,
    y=means_set1,
    name='MAE',
    error_y=dict(type='data', array=std_set1, visible=True),
    marker=dict(color='#646DF9')  # Green color from the previous request
))

# Adding bars for the second set
fig.add_trace(go.Bar(
    x=categories,
    y=means_set2,
    name='PMAE',
    error_y=dict(type='data', array=std_set2, visible=True),
    marker=dict(color='#00CC96')  # Blue color from the previous request
))

# Update layout to match the previous specifications
fig.update_layout(
    barmode='group',
    xaxis_title="Datasets",
    yaxis_title="Lin. Probe Acc. (%)",
    font=dict(size=18),
    showlegend=True,  # Disable global legend
    legend=dict(
            font=dict(size=18),  # Increase legend font size
            x=0.05,  # Position legend inside plot
            y=1,  # Position legend inside plot
            bgcolor='rgba(255,255,255,1.0)'  # Transparent legend background
        ),
    paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
    margin=dict(l=10, r=10, t=40, b=10),  # Adjust margins to fit titles
    width=450,  # Set custom width to fit all subplots
    height=300  # Set custom height
)

# Update x-axis and y-axis settings for all subplots
fig.update_xaxes(showgrid=False, tickfont=dict(size=14))
fig.update_yaxes(showgrid=True, gridcolor="gray", tickfont=dict(size=14))

# Display or save the plot
fig.show() 
fig.write_image("bar_plot_5_subplots_closer_bars.pdf", format='png', scale=2,width=450,height=300)

# %%
