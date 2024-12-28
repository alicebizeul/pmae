#%% 
import os, glob 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#%% 

results_mae = {
    "DermaMNIST":{
        8:{
            50:73.0,
            60:74.0,
            75:74.0,
            80:73.0,
            90:76.0,
        },
    },
    "PathMNIST":{
        8:{
            50:86.6,
            60:88.8,
            75:90.8,
            80:95.5,
            90:94.8,
        },
    },
    "BloodMNIST":{
        8:{
            50:88.0,
            60:93.5,
            75:90.0,
            80:88.0,
            90:84.0,
        },
    },
    "TinyImageNet":{
        8:{
            50:7.9,
            60:10.6,
            75:11.6,
            80:14.4,
            90:15.9,
        },
    },
    "CIFAR10":{
        8:{
            50:38,
            60:39,
            75:41,
            80:50,
            90:39,
        },
    },
}

#%%

# Initialize figure
fig = go.Figure()

# Create radar plot for each dataset in the results_mae
for dataset, values in results_mae.items():
    categories = []
    values_storage = []
    
    # Extract values for theta categories and corresponding values
    for mask_size, mask_values in values.items():
        for theta, val in mask_values.items():
            categories.append(f"{mask_size},{theta}")
            values_storage.append(val)

    values_storage = [x -max(values_storage) for x in values_storage]

    categories += [categories[0]]
    values_storage += [values_storage[0]]
    # Add curves for each dataset
    if values:
        fig.add_trace(go.Scatterpolar(
            r=values_storage,
            theta=categories[:len(values_storage)],
            mode='markers+lines',
            name=f'{dataset}',
            line=dict(width=1.5)
        ))


# Update layout for aesthetics
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
        ),
    ),
    showlegend=True,
    # legend=dict(entrywidth=0.2, # change it to 0.3
    #                           entrywidthmode='fraction',
    #                           orientation='h',
    #                           y=1.2,
    #                           xanchor="center",
    #                           x=0.65)
    legend=dict(
        entrywidth=0.4,  # Adjusting the width of legend entries
        entrywidthmode='fraction',  # Ensure width is a fraction of total width
        orientation='h',  # Horizontal legend
        y=1.2,  # Move legend above the plot
        xanchor="right",  # Align the legend to the center horizontally
        x=0.95,  # Shift the legend slightly to the right
        font=dict(  # Set the font properties for the legend
            size=14,  # Change the font size
            color="black"  # Change the font color (optional)
        ),
    ),
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
    paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent outer background
    margin=dict(l=0, r=0, t=0, b=0)  # Tight cropping around the figure
)

# Save the figure with a transparent background and tight cropping
fig.write_image("./figures/radar_chart.png", scale=1, width=400, height=400)
# Show the plot
fig.show()
# %%
# Updated dataset order
dataset_order = ["CIFAR10", "TinyImageNet", "DermaMNIST", "BloodMNIST", "PathMNIST"]

# Initialize subplot figure with 1 row and 5 columns for the specified dataset order
fig = make_subplots(rows=1, cols=5, subplot_titles=dataset_order)

# Plotly default color palette
plotly_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

# Create the bar plot for each dataset and add it as a subplot
col_index = 1
for index, (dataset, color) in enumerate(zip(dataset_order, plotly_colors)):

    values = results_mae[dataset]
    x_labels = []  # Store x-axis labels (e.g., "8,50")
    y_values = []  # Store the corresponding y-values
    mask_sizes = []  # Store the mask sizes for colors

    # Extract values from the dictionary
    for mask_size, mask_values in values.items():
        for mask_percentage, mae in mask_values.items():
            x_labels.append(f"{mask_percentage}")
            y_values.append(mae)
            mask_sizes.append(mask_size)

    # Calculate the y-axis lower bound (5% below the minimum value)
    min_y, max_y = min(y_values), max(y_values)
    y_axis_lower = min_y -2  # 5% lower than the minimum value
    y_axis_upper = max_y +2 # 5% lower than the minimum value

    # Add bar trace for each dataset as a subplot
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_values,
            # text=[f"{val:.1f}" for val in y_values],  # Display the value at the top of each bar
            # textposition='outside',  # Position the text on top of the bar
            # textangle=90,
            # hoverinfo='x+y+text',
            name=dataset,
            marker=dict(color=plotly_colors[0] if index<2 else plotly_colors[2]),  # Use the specified color from the Plotly palette
        ),
        row=1, col=col_index
    )

    # Update the y-axis for the current subplot to start 5% below the minimum value
    fig.update_yaxes(range=[y_axis_lower, y_axis_upper], row=1, col=col_index)  # Adjust y-axis range dynamically

    # Remove the y-axis label for all but the first plot
    # if col_index != 1:
    #     fig.update_yaxes(showticklabels=False, row=1, col=col_index)

    # Move to the next column for the next dataset
    col_index += 1

# Update layout for aesthetics
fig.update_layout(
    height=300,  # Adjust the height of the figure
    width=1500,  # Adjust the width of the figure to accommodate all subplots
    uniformtext_minsize=28,  # Minimum font size for all text
    uniformtext_mode='show',  # Enforce the minimum size on all text
    showlegend=False,  # Hide the legend to make the subplots cleaner
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
    paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
    margin=dict(l=5, r=5, t=45, b=5),  # Adjust margins to fit titles
    yaxis=dict(showgrid=True, gridcolor='gray'),
    title_font=dict(size=28),
    font=dict(size=28),
    annotations=[dict(font=dict(size=28)) for annotation in fig['layout']['annotations']]
)

ranges={
    "CIFAR10":[35,67],
    "TinyImageNet":[5,24],
    "BloodMNIST":[80,99],
    "DermaMNIST":[70,83],
    "PathMNIST":[80,104]
}

# Customize axes for each subplot
for i in range(1, 6):
    if i==3: fig.update_xaxes(
        title_text="Percentage of patches masked", 
        titlefont=dict(size=28),  # Increase font size for x-axis label
        tickfont=dict(size=28),  # Increase font size for x-axis ticks
        row=1, col=i,
    )
    else:
        fig.update_xaxes(
        titlefont=dict(size=28),  # Increase font size for x-axis label
        tickfont=dict(size=28),  # Increase font size for x-axis ticks
        row=1, col=i
    )

    if i ==1: fig.update_yaxes(title_text="Lin. Acc. (%)", range=ranges[dataset_order[i-1]],  row=1, col=i,titlefont=dict(size=28),tickfont=dict(size=28),gridcolor="gray")
    else: fig.update_yaxes(title_text="", range=ranges[dataset_order[i-1]], row=1, col=i,gridcolor="gray",titlefont=dict(size=28),tickfont=dict(size=28))

# Save the figure with transparent background and tight cropping
fig.write_image("./figures/bar_chart_subplots.pdf", scale=2, width=1500, height=300)

# Show the figure
fig.show()
# %%
