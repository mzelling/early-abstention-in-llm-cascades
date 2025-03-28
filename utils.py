import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression

def plot_pct_change(grids_test, grids_test_no, lambda_cost_grid, lambda_abs_grid, metric_name="loss", filename=None, save_fig=False):
    """ Plot a heatmap of percentage changes. """

    # Set a clean theme for a polished look
    sns.set_theme(context='notebook', style='white', font_scale=1.2)

    cost_tick_indices = np.arange(0, len(lambda_cost_grid), 5)
    cost_tick_labels = [f"{lambda_cost_grid[i]:.1g}" for i in cost_tick_indices]
    abs_tick_indices = np.arange(0, len(lambda_abs_grid), 5)
    abs_tick_labels = [f"{lambda_abs_grid[i]:.2f}" for i in abs_tick_indices]
    abs_labels_subset = [f"{lambda_abs_grid[i]:.1f}" for i in abs_tick_indices]

    fig, ax = plt.subplots( figsize=(8, 6) )

    # First heatmap (Training)
    sns.heatmap(
        100*(
            (np.array(grids_test[metric_name]) - np.array(grids_test_no[metric_name]))
            /(np.array(grids_test_no[metric_name]))
        ),
        annot=False,        
        fmt=".2f",         
        cmap='bwr',
        center=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": f"$\% \Delta$"}, 
        ax=ax
    )
    plt.xticks(ticks=abs_tick_indices, labels=abs_labels_subset)

    ax.set_xticks(cost_tick_indices + 0.5)
    ax.set_xticklabels(cost_tick_labels, rotation=45)
    ax.set_yticks(abs_tick_indices + 0.5)
    ax.set_yticklabels(abs_tick_labels, rotation=90)
    ax.set_xlabel(r"$\lambda_\text{cost}$", fontsize=14, labelpad=8)
    ax.set_ylabel(r"$\lambda_\text{abstention}$", fontsize=14, labelpad=8)

    plt.tight_layout()

    if save_fig:
        plt.savefig(filename, format="pdf", bbox_inches="tight")


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pct_change_v2(
    grids_test,
    grids_test_no,
    lambda_cost_grid,
    lambda_abs_grid,
    metric_name="loss",
    filename=None,
    save_fig=False,
    ax=None
):
    """
    Plot a heatmap of percentage changes on a supplied Axes (if given),
    or create a new Figure/Axes if not provided.
    """

    # Make font sizes a bit larger by default
    # This ensures labels and ticks remain readable in a smaller subplot.
    sns.set_theme(context='notebook', style='white', font_scale=1.4)

    # If no Axes object is provided, create one (standalone usage).
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        own_fig = True

    # Calculate the percentage change
    percentage_change = 100 * (
        (np.array(grids_test[metric_name]) - np.array(grids_test_no[metric_name]))
        / np.array(grids_test_no[metric_name])
    )

    # Define tick indices and labels for cost/abstention
    cost_tick_indices = np.arange(0, len(lambda_cost_grid), 5)
    cost_tick_labels = [f"{lambda_cost_grid[i]:.1g}" for i in cost_tick_indices]

    abs_tick_indices = np.arange(0, len(lambda_abs_grid), 5)
    abs_tick_labels = [f"{lambda_abs_grid[i]:.2f}" for i in abs_tick_indices]
    # You can also simplify if desired, e.g. one decimal place:
    # abs_tick_labels = [f"{lambda_abs_grid[i]:.1f}" for i in abs_tick_indices]

    # Plot the heatmap
    heatmap = sns.heatmap(
        percentage_change,
        annot=False,
        fmt=".2f",
        cmap="bwr",
        center=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label":  "$\\% \\Delta$", "shrink": 0.8, 
                  "aspect": 30},
        ax=ax
    )

    # Grab the colorbar object
    cbar = heatmap.collections[0].colorbar

    # Now you can rotate its label any way you want
    cbar.set_label("$~~~\\% \\Delta$", rotation=0)

    # Manually set tick labels and positions
    ax.set_xticks(cost_tick_indices + 0.5)
    ax.set_xticklabels(cost_tick_labels, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(abs_tick_indices + 0.5)
    ax.set_yticklabels(abs_tick_labels, rotation=0, fontsize=12)

    ax.set_xlabel(r"$\boldsymbol{\lambda_{\text{c}}}$", fontsize=18, labelpad=10)
    ax.set_ylabel(r"$\boldsymbol{\lambda_{\text{a}}}$  ", fontsize=18, labelpad=10).set_rotation(0)
    
    # If we created our own figure, apply tight_layout and possibly save/show it
    if own_fig:
        plt.tight_layout()
        if save_fig and filename is not None:
            plt.savefig(filename, format="pdf", bbox_inches="tight")
        else:
            plt.show()



def flatten_data(data):
    rows = []
    
    # Iterate through the outer dictionary (cascade names)
    for cascade, tasks in data.items():
        # Iterate through each task and its metrics
        for task, metrics in tasks.items():
            # Create a row with cascade and task names, plus all metrics
            row = {
                'cascade': cascade,
                'task': task,
                **metrics  # Unpack all metrics into the row
            }
            rows.append(row)
    
    return rows


def format_cascade_name(name):
    QWEN_OAI_MODELS = ['4o-Mini', 'Q32B', 'Q72B', '4o']
    LLAMA_MODELS = ['L1B', 'L3B', 'L8B', 'L70B', 'L405B']

    model_idx = [ int(x) for x in name.split("chain")[-1] ]

    chain_name = "_".join(name.split("_")[:-1])
    if chain_name == "qwen_oai":
        return QWEN_OAI_MODELS[model_idx[0]] + ' -> ' + QWEN_OAI_MODELS[model_idx[1]]
    elif chain_name == "llama":
        return LLAMA_MODELS[model_idx[0]] + ' -> ' + LLAMA_MODELS[model_idx[1]]


def format_metric_name(name, view='test'):
    postfix = f" on the {view} set" if view=='test' else f" on the {view}ing set"
    if name=="cost":
        return "expected cost" + postfix
    elif name=="abstention":
        return "abstention rate" + postfix
    elif name=="error":
        return "error rate" + postfix
    elif name =="loss":
        return "overall loss" + postfix
    

def generate_latex_table_3cols(
        df, caption, label, value_cols=['x', 'y', 'z'],
        value_cols_labels=None,
        cascade_col='Cascade', 
        benchmark_col='Benchmark', 
        bold_smaller=None,  # None = no comparison bolding, True = bold smaller, False = bold larger
        bold_negative=False,  # Whether to bold negative values in the third column
        decimal_places=3):  # Number of decimal places to round to
    """
    Generate a LaTeX table from a long-format data frame with three values per benchmark.
    Preserves the original order of cascades as they appear in the input data frame.
    
    Parameters
    ----------
    df : pandas DataFrame
        Long-format DataFrame with columns for:
        - cascade_col: The cascade information
        - benchmark_col: The benchmark name
        - value_cols: List of columns containing the three values per benchmark
    caption : str
        Table caption
    label : str
        Table label for LaTeX references
    value_cols : list
        Names of the columns containing the three values per benchmark
    value_cols_labels : list
        Labels for the value columns in the table
    cascade_col : str
        Name of the column containing cascade information
    benchmark_col : str
        Name of the column containing benchmark names
    bold_smaller : bool or None
        If True, bold the smaller of first two values
        If False, bold the larger of first two values
        If None, no comparison bolding is applied
    bold_negative : bool
        If True, bold negative values in the third column
    
    Returns
    -------
    str : code for LaTeX table
    """
    if len(value_cols) != 3:
        raise ValueError("Exactly three value columns must be provided")

    # Get unique cascades in their original order
    cascades = df[cascade_col].unique()
    
    # Get unique benchmarks and sort them
    benchmarks = sorted(df[benchmark_col].unique())
    
    # Start building the table
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\small")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    
    # Calculate number of columns: 1 for cascade + 3 for each benchmark + 3 for average
    n_cols = 1 + len(benchmarks) * 3 + 3
    latex.append("\\begin{adjustbox}{max width=\\textwidth}")
    latex.append("\\begin{tabular}{l" + "rrr" * (len(benchmarks) + 1) + "}")
    latex.append("\\toprule")
    
    def escape_latex(text):
        """Escape LaTeX special characters, particularly underscores"""
        return text.replace('_', '\\_')
    
    # Header row 1 - Cascade and Benchmarks
    headers1 = ["\\multirow{2}{*}{\\textbf{" + escape_latex(cascade_col) + "}}"]
    for benchmark in benchmarks:
        headers1.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{escape_latex(benchmark)}}}}}")
    headers1.append("\\multicolumn{3}{c}{\\textbf{Average}}")
    latex.append(" & ".join(headers1) + " \\\\")
    
    # Header row 2 - Value types (x, y, z)
    if value_cols_labels is None:
        value_cols_labels = [escape_latex(col) for col in value_cols]

    headers2 = [""]  # Empty cell for Cascade column
    value_headers = [f"\\textbf{{{col}}}" for col in value_cols_labels]
    for _ in range(len(benchmarks) + 1):  # +1 for Average
        headers2.extend(value_headers)
    latex.append(" & ".join(headers2) + " \\\\")
    
    # Add midrule
    latex.append("\\midrule")
    
    # Prepare pivot tables for each value column
    pivot_dfs = {}
    for val_col in value_cols:
        pivot_df = df.pivot(index=cascade_col, columns=benchmark_col, values=val_col)
        pivot_df = pivot_df.reindex(cascades)
        pivot_df['Average'] = pivot_df.mean(axis=1)
        pivot_dfs[val_col] = pivot_df
    
    # Calculate column averages
    avg_series = {val_col: pivot_dfs[val_col].mean() for val_col in value_cols}
    
    def format_value_triple(val1, val2, val3):
        """Format a triple of values with appropriate bolding"""
        formatted = []
        
        # Handle first two values based on bold_smaller
        if bold_smaller is None:
            formatted.extend([
                f"{val1:.{decimal_places}f}",
                f"{val2:.{decimal_places}f}"
            ])
        else:
            if bold_smaller == (val1 < val2):
                formatted.extend([
                    f"\\textbf{{{val1:.{decimal_places}f}}}",
                    f"{val2:.{decimal_places}f}"
                ])
            else:
                formatted.extend([
                    f"{val1:.{decimal_places}f}",
                    f"\\textbf{{{val2:.{decimal_places}f}}}"
                ])
        
        # Handle third value based on bold_negative
        val3_formatted = f"{val3:.{decimal_places}f}"
        if bold_negative and val3 < 0:
            val3_formatted = f"\\textbf{{{val3_formatted}}}"
        formatted.append(val3_formatted)
        
        return ["{" + val + "}" for val in formatted]
    
    # Data rows
    for cascade in cascades:
        row_values = [f"\\textbf{{{escape_latex(cascade)}}}"]
        
        # Add values for each benchmark
        for benchmark in benchmarks + ['Average']:
            vals = [pivot_dfs[val_col].loc[cascade, benchmark] for val_col in value_cols]
            row_values.extend(format_value_triple(*vals))
        
        latex.append(" & ".join(row_values) + " \\\\")
    
    # Add average row
    latex.append("\\midrule")
    avg_row = ["\\textbf{" + escape_latex("Average") + "}"]
    
    for benchmark in benchmarks + ['Average']:
        vals = [avg_series[val_col][benchmark] for val_col in value_cols]
        avg_row.extend(format_value_triple(*vals))
    
    latex.append(" & ".join(avg_row) + " \\\\")
    
    # Close the table
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{adjustbox}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def plot_precision_recall_curve(model, X_test, y_test, plot=False, pos_label=1, figsize=(10, 6)):
    """
    Compute and plot precision-recall curve for a fitted logistic regression model.
    
    Parameters:
    -----------
    model : sklearn model (e.g., LogisticRegression)
        The fitted classification model with a predict_proba method
    X_test : array-like
        Test features
    y_test : array-like
        True labels for test data
    plot : bool
        Whether to plot the precision-recall curve right now.
    pos_label : int, default=1
        The label of the positive class
    figsize : tuple, default=(10, 6)
        Figure size for the plot
        
    Returns:
    --------
    precision : array
        Precision values
    recall : array
        Recall values
    thresholds : array
        Threshold values used to compute precision and recall
    """
    # Get predicted probabilities for the positive class
    y_score = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision and recall for different thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_score, pos_label=pos_label)
    
    # Calculate average precision score
    ap = average_precision_score(y_test, y_score, pos_label=pos_label)
    
    if plot:
        # Create the precision-recall curve plot
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {ap:.3f})')
        
        # Add a reference line for random classifier
        plt.plot([0, 1], [sum(y_test == pos_label) / len(y_test)] * 2, 
                color='red', linestyle='--', label='Random classifier')
        
        # Set plot aesthetics
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
    
    return precision, recall, thresholds