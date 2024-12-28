import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os

# Example loss_tracking data
# loss_tracking = [...]

def plot_loss(loss,name_loss,save_dir,name_file=""):
    # Create a subplot with 1 row and 2 columns
    os.makedirs(save_dir, exist_ok=True)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Linear Scale', 'Log Scale'))

    # First subplot (Linear scale)
    fig.add_trace(
        go.Scatter(y=loss, mode='lines', name='Linear Scale'),
        row=1, col=1
    )

    # Second subplot (Log scale)
    fig.add_trace(
        go.Scatter(y=loss, mode='lines', name='Log Scale'),
        row=1, col=2
    )

    # Set the y-axis of the second subplot to log scale
    fig.update_yaxes(type="log", row=1, col=2)

    # Set common y-axis label for both plots
    fig.update_yaxes(title_text=name_loss, row=1, col=1)

    # Update layout for better presentation
    fig.update_layout(
        height=500,  # Adjust the figure height
        width=1000,  # Adjust the figure width
        title_text="Loss Tracking",
    )

    # Save the figure as a PNG file
    output_path = os.path.join(save_dir, f"loss{name_file}.png")
    fig.write_image(output_path, scale=2)

def plot_performance(x,y,save_dir,name=""):
    os.makedirs(save_dir, exist_ok=True)
    # Create a Plotly figure
    fig = go.Figure()

    # Add a line plot
    fig.add_trace(go.Scatter(
        x=[a+1 for a in list(x)],  # X-axis: class labels
        y=list(y),  # Y-axis: accuracy values
        mode='lines+markers',
        name="Downstream prediction"
    ))

    # Set axis labels and title
    fig.update_layout(
        title='Model Performance',
        xaxis_title='Epochs',
        yaxis_title='Accuracy',
        font=dict(size=15)
    )

    # Save the figure as a PNG file
    output_path = os.path.join(save_dir,f"performance_{name}.png")
    fig.write_image(output_path, scale=2)




