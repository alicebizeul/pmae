#%% 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Data
cifar10 = {
    'mae':{
        60: 260,
        75: 255,
        80: 260,
        90: 255,
    },
    'pmae':{
        0.0: 300
    }
}

tinyimagenet = {
    'mae':{
        60: 1740,
        75: 1650,
        80: 1650,
        90: 1590,
    },
    'pmae':{
        0.0: 2250
    }
}

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('CIFAR-10', 'TinyImageNet'))

# CIFAR-10 MAE plot
x_cifar = list(cifar10['mae'].keys())
y_cifar = list(cifar10['mae'].values())
pmae_cifar = cifar10['pmae'][0.0]

fig.add_trace(go.Scatter(
    x=x_cifar, y=y_cifar, mode='lines+markers',
    name='MAE',
    line=dict(color='blue'),
), row=1, col=1)

# Add dashed horizontal line for CIFAR-10 PMAE
fig.add_trace(go.Scatter(
    x=[min(x_cifar), max(x_cifar)], y=[pmae_cifar, pmae_cifar],
    mode='lines',
    name='PMAE',
    line=dict(color='blue', dash='dash')
), row=1, col=1)

# TinyImageNet MAE plot
x_tiny = list(tinyimagenet['mae'].keys())
y_tiny = list(tinyimagenet['mae'].values())
pmae_tiny = tinyimagenet['pmae'][0.0]

fig.add_trace(go.Scatter(
    x=x_tiny, y=y_tiny, mode='lines+markers',
    name='MAE',
    line=dict(color='#636EFA'),
), row=1, col=2)

# Add dashed horizontal line for TinyImageNet PMAE
fig.add_trace(go.Scatter(
    x=[min(x_tiny), max(x_tiny)], y=[pmae_tiny, pmae_tiny],
    mode='lines',
    name='PMAE',
    line=dict(color='#636EFA', dash='dash')
), row=1, col=2)

# Update layout with tight margins and save the figure
fig.update_layout(
    showlegend=True,
    height=400,
    width=800,
    margin=dict(l=10, r=10, t=30, b=10)  # Reduce margins around the plot
)

# Add axis titles
fig.update_xaxes(title_text="Masking Ratio", row=1, col=1)
fig.update_yaxes(title_text="Training Time (min)", row=1, col=1)
fig.update_xaxes(title_text="Masking Ratio", row=1, col=2)

# Save the figure as a PNG file with a resolution scale of 2
fig.write_image("mae_pmae_plot.png", scale=2)

# Show the figure
fig.show()

# %%
