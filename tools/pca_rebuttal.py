import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import plotly.colors as pc

# Generate data for y = x^2 + epsilon
np.random.seed(42)
x = np.linspace(-10, 10, 1000)
epsilon = np.random.normal(0, 5.0, x.shape)
y1 = x**2 + epsilon
data1 = np.vstack((x, y1)).T
data1 = (data1 - np.mean(data1, axis=0)) / np.std(data1, axis=0)

# Perform PCA for the first dataset
pca1 = PCA(n_components=2)
pca1.fit(data1)
principal_components1 = pca1.components_
projected_data1 = pca1.transform(data1)

# Generate data for y = 3x + epsilon
epsilon = np.random.normal(0, 5.0, x.shape)
y2 = 3 * x + epsilon
data2 = np.vstack((x, y2)).T
data2 = (data2 - np.mean(data2, axis=0)) / np.std(data2, axis=0)

# Perform PCA for the second dataset
pca2 = PCA(n_components=2)
pca2.fit(data2)
principal_components2 = pca2.components_
projected_data2 = pca2.transform(data2)

# Generate data for y = 3x + epsilon
epsilon = np.random.normal(0, 30.0, x.shape)
y3 = 3 * x + epsilon
data3 = np.vstack((x, y3)).T
data3 = (data3 - np.mean(data3, axis=0)) / np.std(data3, axis=0)

# Perform PCA for the second dataset
pca3 = PCA(n_components=2)
pca3.fit(data3)
principal_components3 = pca3.components_
projected_data3 = pca3.transform(data3)

# Use default Plotly color palette
palette = pc.qualitative.Plotly
colors = {
    "data": palette[0],
    "pc": [palette[1], palette[2]],
    "projected": palette[3]
}

# Create subplots
fig = make_subplots(rows=1, cols=3, subplot_titles=(r"$y = x^2 + z, z \sim \mathcal{N(0,5)}$", r"$y = 3x + z, z \sim \mathcal{N(0,5)}$", r"$y = 3x + z, z \sim \mathcal{N(0,30)}$"))

# Add Original Data to both subplots
fig.add_trace(
    go.Scatter(
        x=data1[:, 0],
        y=data1[:, 1],
        mode='markers',
        name='Original Data',
        marker=dict(size=5, color=colors["data"]),
        legendgroup="data",
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=data2[:, 0],
        y=data2[:, 1],
        mode='markers',
        name='Original Data',
        marker=dict(size=5, color=colors["data"]),
        legendgroup="data",
        showlegend=False
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=data3[:, 0],
        y=data3[:, 1],
        mode='markers',
        name='Original Data',
        marker=dict(size=5, color=colors["data"]),
        legendgroup="data",
        showlegend=False
    ),
    row=1, col=3
)

# Add Principal Components to both subplots
for i, (component1, component2, component3) in enumerate(zip(principal_components1, principal_components2, principal_components3)):
    fig.add_trace(
        go.Scatter(
            x=[-5 * component1[0], 5 * component1[0]],
            y=[-5 * component1[1], 5 * component1[1]],
            mode='lines',
            name=f'PC{i + 1}',
            line=dict(width=2, dash='dash', color=colors["pc"][i]),
            legendgroup=f"pc{i + 1}"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[-5 * component2[0], 5 * component2[0]],
            y=[-5 * component2[1], 5 * component2[1]],
            mode='lines',
            name=f'PC{i + 1}',
            line=dict(width=2, dash='dash', color=colors["pc"][i]),
            legendgroup=f"pc{i + 1}",
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[-5 * component3[0], 5 * component3[0]],
            y=[-5 * component3[1], 5 * component3[1]],
            mode='lines',
            name=f'PC{i + 1}',
            line=dict(width=2, dash='dash', color=colors["pc"][i]),
            legendgroup=f"pc{i + 1}",
            showlegend=False
        ),
        row=1, col=3
    )

# Add Projected Data to both subplots
fig.add_trace(
    go.Scatter(
        x=projected_data1[:, 0],
        y=projected_data1[:, 1],
        mode='markers',
        name='Data projected on pcs',
        marker=dict(size=5, symbol='cross', color=colors["projected"]),
        legendgroup="projected",
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=projected_data2[:, 0],
        y=projected_data2[:, 1],
        mode='markers',
        name='Data projected on pcs',
        marker=dict(size=5, symbol='cross', color=colors["projected"]),
        legendgroup="projected",
        showlegend=False
    ),
    row=1, col=2
)


fig.add_trace(
    go.Scatter(
        x=projected_data3[:, 0],
        y=projected_data3[:, 1],
        mode='markers',
        name='Data projected on pcs',
        marker=dict(size=5, symbol='cross', color=colors["projected"]),
        legendgroup="projected",
        showlegend=False
    ),
    row=1, col=3
)

# Update layout
fig.update_layout(
    showlegend=True,
    legend=dict(
        orientation="h",  # Horizontal legend
        yanchor="bottom",  # Anchor legend at the bottom
        xanchor="center",  # Center the legend horizontally
        x=0.5,             # Position legend at the center
        y=-0.2,            # Place legend below the plot
        traceorder="grouped",
        font=dict(size=16),
    ),
    width=1400,
    height=500,  # Adjust height to accommodate legend space
    margin=dict(
        l=10,  # Left margin
        r=10,  # Right margin
        t=30,  # Top margin
        b=50   # Bottom margin to make space for the legend
    ),
)

fig.write_image("reviewer2_default_palette.png")
