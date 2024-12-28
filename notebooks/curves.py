#%%
import plotly.graph_objects as go
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Data for each dataset
epochs = [200, 400, 600, 800]

# Tiny dataset values
tiny_pmae_og = [19.1, 18.5, 18.0, 17.4]
tiny_pmae = [22.3,22.3,22.6,22.5]
tiny_mae = [17.7, 17.1, 16.3, 15.5]

# CIFAR dataset values: loss original
cifar_pmae_og = [51.8, 53.4, 54.3, 55.1]
cifar_pmae = [52.4,56.5,58.3,59.0]
cifar_mae = [49.8, 50.4, 50.5, 50.7]

# Blood dataset values
blood_pmae_og = [89.0, 89.5, 90.5, 91.0]
blood_pmae = [89.0, 92.0, 95.0, 95.3]
blood_mae = [74.7, 75.2, 77.9, 77.6]

# Blood dataset values
derma_pmae_og = [75.3,76.7,77.2,77.4]
derma_pmae = [75.2,77.0,78.3,78.6]
derma_mae = [70.8,72.3,73.5,73.5]

# Path
path_pmae_0g = [94.5,94.4,94.7,94.7]
path_pmae = [95.9,96.6,96.6,96.6]
path_mae = [86.1,86.7,86.3,86.1]

# Function to plot the data using plotly
def plot_data_plotly(epochs, pmae, mae, title, position):
    fig = go.Figure()
    
    # Add MAE curve
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=mae, 
        mode='lines+markers', 
        name='MAE', 
        line=dict(color='#646DF9', width=3),  # Pastel blue
        marker=dict(symbol='square')
    ))

    # Add PMAE curve
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=pmae, 
        mode='lines+markers', 
        name='PMAE', 
        line=dict(color='#00CC96', width=3),  # Pastel green
        marker=dict(symbol='circle')
    ))
    
    
    # Update layout
    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Lin. Probe Acc. (%)",
        font=dict(size=18),  # Font size for all text
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        width=450,  # Adjust the width
        height=300,  # Adjust the height
        showlegend=True,
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
        margin=dict(l=5, r=5, t=30, b=5),  # Adjust margins to fit titles
        legend=dict(
            font=dict(size=18),  # Increase legend font size
            x=0.8 if position == "right" else 0.02,  # Position legend inside plot (adjust x)
            y=0.98,  # Position legend inside plot (adjust y)
            bgcolor='rgba(255,255,255,1.0)',  # Set background color of the legend (optional)
        ),        xaxis=dict(showgrid=True, tickfont=dict(size=16), gridcolor='gray'),  # X-axis ticks and grid
        yaxis=dict(showgrid=True, tickfont=dict(size=16), gridcolor='gray')   # Y-axis ticks and grid
    )
    
    
    fig.write_image(f"./{title}.pdf", scale=2, width=450, height=300)
    # fig.show()

#%%
# Plot for Tiny Dataset
plot_data_plotly(epochs, tiny_pmae, tiny_mae, "Tiny Dataset",position="right")

#%%
# Plot for CIFAR Dataset
plot_data_plotly(epochs, cifar_pmae, cifar_mae, "CIFAR Dataset",position="left")

#%%
# Plot for Blood Dataset
plot_data_plotly(epochs, blood_pmae, blood_mae, "Blood Dataset",position="left")

#%%
# Plot for Blood Dataset
plot_data_plotly(epochs, derma_pmae, derma_mae, "Derma Dataset",position="left")


#%%
# Plot for Path 
plot_data_plotly(epochs, path_pmae, path_mae, "Path Dataset",position="left")
# %%
