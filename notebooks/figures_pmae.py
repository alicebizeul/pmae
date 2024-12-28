#%% 
import os, glob 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#%% 
# Loss Original
results_mae = {
    "DermaMNIST":{
        8:{
            10: 78.2,
            20: 78.5,
            30: 76.4,
            40: 73.6,
            50: 71.4,
            60: 76.3,
            70: 71.5,
            80: 71.5,
            90: 71.6
        },
    },
    "PathMNIST":{
        8:{
            10: 96.9,
            20: 94.5,
            30: 94.8,
            40: 95.0,
            50: 88.8,
            60: 81.7,
            70: 83.1,
            80: 84.0,
            90: 84.4,
        },
    },
    "BloodMNIST":{
        8:{
            10: 92.5,
            20: 93.5,
            30: 93.8,
            40: 88.2,
            50: 85.0,
            60: 85.2,
            70: 83.6,
            80: 84.9,
            90: 86.0,
        },
    },
    "TinyImageNet":{
        8:{
            10: 20.7,
            20: 20.3,
            30: 16.3,
            40: 17.4,
            50: 15.0,
            60: 14.4,
            70: 14.4,
            80: 12.8,
            90: 11.6,
        },
    },
    "CIFAR10":{
        8:{
            10: 59.5,
            20: 58.8,
            30: 59.5,
            40: 58.2,
            50: 57.3,
            60: 53.1,
            70: 52.7,
            80: 53.6,
            90: 50.0,
        },
    },
}

# # Loss A 
# results_mae = {
#     "DermaMNIST":{
#         8:{
#             10: 78.2,
#             20: 78.6,
#             30: 78.2,
#             40: 77.0,
#             50: 78.2,
#         },
#     },
#     "PathMNIST":{
#         8:{
#             10: 96.5,
#             20: 94.6,
#             30: 94.0,
#             40: 92.0,
#             50: 90.2,
#         },
#     },
#     "BloodMNIST":{
#         8:{
#             10: 93.7,
#             20: 93.3,
#             30: 95.3,
#             40: 94.5,
#             50: 92.4,
#         },
#     },
#     "TinyImageNet":{
#         8:{
#             10: 13.5,
#             20: 22.5,
#             30: 21.1,
#             40: 16.7,
#             50: 22.2,
#         },
#     },
#     "CIFAR10":{
#         8:{
#             10: 59.0,
#             20: 57.0,
#             30: 57.8,
#             40: 55.7,
#             50: 55.3,
#         },
#     },
# }
#%%

# # Initialize figure
# fig = go.Figure()

# # Create radar plot for each dataset in the results_mae
# for dataset, values in results_mae.items():
#     categories = []
#     values_storage = []
    
#     # Extract values for theta categories and corresponding values
#     for mask_size, mask_values in values.items():
#         for theta, val in mask_values.items():
#             categories.append(f"{mask_size},{theta}")
#             values_storage.append(val)

#     values_storage = [x -max(values_storage) for x in values_storage]

#     categories += [categories[0]]
#     values_storage += [values_storage[0]]
#     # Add curves for each dataset
#     if values:
#         fig.add_trace(go.Scatterpolar(
#             r=values_storage,
#             theta=categories[:len(values_storage)],
#             mode='markers+lines',
#             name=f'{dataset}',
#             line=dict(width=1.5)
#         ))


# # Update layout for aesthetics
# fig.update_layout(
#     polar=dict(
#         radialaxis=dict(
#             visible=True,
#         ),
#     ),
#     showlegend=True,
#     # legend=dict(entrywidth=0.2, # change it to 0.3
#     #                           entrywidthmode='fraction',
#     #                           orientation='h',
#     #                           y=1.2,
#     #                           xanchor="center",
#     #                           x=0.65)
#     legend=dict(
#         entrywidth=0.4,  # Adjusting the width of legend entries
#         entrywidthmode='fraction',  # Ensure width is a fraction of total width
#         orientation='h',  # Horizontal legend
#         y=1.2,  # Move legend above the plot
#         xanchor="right",  # Align the legend to the center horizontally
#         x=0.95,  # Shift the legend slightly to the right
#         font=dict(  # Set the font properties for the legend
#             size=14,  # Change the font size
#             color="black"  # Change the font color (optional)
#         ),
#     ),
#     plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
#     paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent outer background
#     margin=dict(l=0, r=0, t=0, b=0)  # Tight cropping around the figure
# )

# # Save the figure with a transparent background and tight cropping
# fig.write_image("./figures/radar_chart.pdf", scale=1, width=400, height=400)
# # Show the plot
# fig.show()
# %%
# Updated dataset order
dataset_order = ["CIFAR10", "TinyImageNet", "DermaMNIST", "BloodMNIST", "PathMNIST"]

# Initialize subplot figure with 1 row and 5 columns for the specified dataset order
fig = make_subplots(rows=1, cols=5)

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
    for mask_size, mask_values in sorted(values.items()):
        for mask_percentage, mae in sorted(mask_values.items()):
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
    # annotations=[dict(font=dict(size=20)) for annotation in fig['layout']['annotations']]
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
        title_text="Percentage of variance masked", 
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

    if i ==1: fig.update_yaxes(title_text="Lin. Acc. (%)", range=ranges[dataset_order[i-1]], row=1,titlefont=dict(size=28),tickfont=dict(size=28),  col=i,gridcolor="gray")
    else: fig.update_yaxes(title_text="", range=ranges[dataset_order[i-1]], row=1, col=i,gridcolor="gray",titlefont=dict(size=28),tickfont=dict(size=28))

# Save the figure with transparent background and tight cropping
fig.write_image("./figures/bar_chart_subplots_p.pdf", scale=2, width=1500, height=300)

# Show the figure
fig.show()
# %%
