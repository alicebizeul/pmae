#%%

import os, glob 
import pandas as pd
from yaml import safe_load
import plotly.express as px
import plotly.graph_objects as go

# Function to flatten the dictionary, keeping each level in a separate column
def flatten_dict_to_columns(d, parent_key=(), sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + (k,)  # Build the key tuple
        if isinstance(v, dict):
            items.extend(flatten_dict_to_columns(v, new_key, sep=sep))
        else:
            items.append(new_key + (v,))
    return items
root_path = "/cluster/project/sachan/callen/mae_alice/outputs/callen"

with open('/cluster/home/callen/projects/mae/config/user/callen_euler.yaml', 'r') as f:
    mydict =safe_load(f)["checkpoint_folders"]

# Flatten the dictionary, keeping each level of depth as a separate column
flattened_data = flatten_dict_to_columns(mydict)

# Convert to a pandas DataFrame
df = pd.DataFrame(flattened_data)

#%%
# Fill in column names based on the maximum depth of the nested dictionary
max_depth = df.shape[1] - 1  # Last column is the value
df.columns = ["vit","dataset","setting","strategy","threshold","patch size","folder"]

df["threshold"]  = df["threshold"].map(lambda x: int(x.split("_")[-1]) if (x!="_None" and x != "_") else None)
# df["patch size"] = df["patch size"].map(lambda x: int(x.split("_")[-1]) if (x is not None and x != "_") else None)

#%%
df_results = []
for i, row in df.iterrows():
    if row["folder"] is not None:
        for eval_type in ["lin","knn"]:
            path = os.path.join(root_path,row["folder"])
            result_files = glob.glob(os.path.join(path,f"performance_final_*_{eval_type}.csv"))
            for r_f in result_files:
                print(r_f)
                epoch = int(r_f.split("/")[-1].split(".csv")[0].split("_")[-2])
                performances = pd.read_csv(r_f)
                result = max(performances["Test Accuracy"].values)
                result_dict = {"epoch":epoch,"accuracy":result,"eval_type":eval_type}
                result_dict.update(row)
                df_results.append(result_dict)
df_results=pd.DataFrame(df_results)
df_results['strategy_setting_model'] = df_results['strategy'] + '_' + df_results['setting']

# %%
# Views 
df_results[(df_results["dataset"]=="cifar10") & (df_results["epoch"]==800)]

#%%
# Take best runs for each approach
# Step 1: Identify the best "strategy"/"setting" for each dataset at epoch 800
best_rows = df_results[(df_results['epoch'] == 800) & (df_results['eval_type']=='lin')].groupby(['dataset','strategy','setting','vit']).apply(lambda x: x.loc[x['accuracy'].idxmax()])
#%%
# Step 2: Filter the original dataframe to keep only the rows that match the best "strategy"/"setting" for each dataset
df_results_filtered_lin = pd.concat([
    df_results[(df_results['eval_type'] == 'lin')&(df_results['dataset'] == row['dataset'])& (df_results['vit'] == row['vit']) & (df_results['strategy'] == row['strategy']) & (df_results['setting'] == row['setting']) & ((df_results['threshold'] == row['threshold']) | (df_results['threshold'] != df_results["threshold"]))]
    for _, row in best_rows.iterrows()
])
df_results_filtered_knn = pd.concat([
    df_results[(df_results['eval_type'] == 'knn')&(df_results['dataset'] == row['dataset'])& (df_results['vit'] == row['vit']) & (df_results['strategy'] == row['strategy']) & (df_results['setting'] == row['setting']) & ((df_results['threshold'] == row['threshold']) | (df_results['threshold'] != df_results["threshold"]))]
    for _, row in best_rows.iterrows()
])


#%%

def create_accuracy_plots(dataframe):
    default_colors = px.colors.qualitative.Plotly
    strategy_settings = dataframe['strategy_setting'].unique()
    color_mapping = {strategy: default_colors[i % len(default_colors)] for i, strategy in enumerate(strategy_settings)}

    datasets = dataframe['dataset'].unique()
    models = dataframe['vit'].unique()
    figures = {}

    # Loop through each dataset
    for dataset in datasets:
        figures[dataset]={}
        for model in models:
            df_dataset = dataframe[(dataframe['dataset'] == dataset) & (dataframe['vit'] == model)]
            # Initialize an empty figure for each dataset
            fig = go.Figure()

            # Loop through each strategy_setting for the dataset and add its curve to the figure
            for strategy_setting in df_dataset['strategy_setting'].unique():
                df_strategy = df_dataset[df_dataset['strategy_setting'] == strategy_setting].sort_values(by='epoch')
                print(dataset,model,df_strategy['epoch'].values,df_strategy['accuracy'].values)

                # Add the line for the current strategy_setting
                fig.add_trace(go.Scatter(
                    x=df_strategy['epoch'].values,
                    y=df_strategy['accuracy'].values,
                    mode='lines+markers',
                    name=strategy_setting,
                    line=dict(color=color_mapping[strategy_setting])
                ))

            # Customize aesthetics for each figure
            fig.update_layout(
                title=f'Accuracy over Epochs for {dataset}',
                xaxis_title='Epoch',
                yaxis_title='Accuracy (%)',
                template="plotly_white",
                title_x=0.5,
                font=dict(size=14),
                legend_title="Strategy Setting",
            )

            figures[dataset][model] = fig

    return figures

# Create the figures for each dataset
figures = create_accuracy_plots(df_results_filtered)

#%%
# Display the first figure for preview
figures["cifar10"]["vit-t"].show()
figures["cifar10"]["vit-b"].show()

# %%
figures["tinyimagenet"].show()

# %%
figures["dermamnist"].show()

# %%
figures["pathmnist"].show()

# %%
figures["bloodmnist"].show()

# %%
