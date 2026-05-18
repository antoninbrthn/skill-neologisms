"""Main evaluation script for WandB runs."""

import argparse
import os
import pandas as pd
from typing import List
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from src.plot_utils import COLS, format_fig
from src.wandb_utils import load_wandb_data
from sequence_map_experiment.config import FIGS_DIR, RESULTS_DIR


def rename_op(op):
    op = op.strip("[]").replace("_", "-")
    op = op.replace("SHIFT-RIGHT", "SHIFT").replace("INVERT-POLARITY", "INV-POL")
    op = op.replace("POLARITY", "POL").replace("REVERSE", "REV")
    return op


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "cm"


# set to default seaborn color palette
sns.set_palette(sns.color_palette([v for k, v in COLS.items()]))
custom_palette = sns.color_palette([v for k, v in COLS.items()])
alternate_order = [
    "blue",
    "blue_l",
    "orange",
    "orange_l",
    "green",
    "green_l",
    "red",
    "red_l",
    "purple",
    "purple_l",
    "almond",
    "almond_l",
]
custom_palette_alternate = sns.color_palette([COLS[k] for k in alternate_order])
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[v for k, v in COLS.items()])


def plot_neologisms_vs_baselines(tags: List[str] = None, reload_cache: bool = True):
    """Load models from wandb tags and evaluate on a test dataset.

    Args:
        tags: List of wandb tags to load
        project_name: WandB project name
        reload_cache: If True, reload from wandb API even if cache exists
        group_by: List of columns to group by
    """
    # Load data
    df_configs, df_results = load_wandb_data(tags=tags, reload_cache=reload_cache)

    # Group results
    df_merged = df_results.merge(df_configs.reset_index(), on="run_id", how="left", suffixes=("", "_config"))
    test_3op_cols = [col for col in df_merged.columns if "3op/" in col]
    test_2op_cols = [col for col in df_merged.columns if "2op/" in col]
    group_by = ["tags", "dataset.skill_op", "dataset.test_op", "epoch"]
    skill_ops = ["[SHIFT_RIGHT]", "[INVERT_POLARITY]"]
    dt_plot = df_merged.groupby(group_by)[test_2op_cols + test_3op_cols].mean().reset_index()
    dt_plot = dt_plot[dt_plot["dataset.skill_op"].isin(skill_ops)]
    dt_plot["tags_str"] = dt_plot["tags"].apply(lambda x: "_".join(x.split("_")).lower())
    # plot acc from last epoch
    dt_plot = dt_plot[dt_plot["epoch"] == dt_plot["epoch"].max()]
    dt_plot["tags_str"].unique()
    dt_plot["model_name"] = dt_plot["tags_str"].replace(
        {
            "baseline,lora": "LoRA",
            "baseline,prompt-tuning": "PT",
            "skill-tokens": "Skill Neologisms",
        }
    )

    # Plotting
    fs = 22
    is_save = True

    def plot_barplot(
        dt_plot,
        k=2,
        # Layout parameters
        figsize=(16, 4),
        bar_width=0.125,
        bar_spacing=0.00,
        # Font sizes
        axis_fontsize=fs,
        tick_fontsize=fs - 4,
        legend_fontsize=fs - 2,
        skill_label_fontsize=fs + 2,
        column_header_fontsize=fs + 6,
        palette=custom_palette_alternate,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
        # Legend
        legend_loc="upper center",
        legend_ncol=None,  # auto-detect
        legend_bbox=(0.5, 0.1),
        # Grid
        show_grid=False,
        grid_alpha=0.3,
        # Spines
        spine_linewidth=1,
        # Average lines
        show_avg=True,
        avg_line_color="black",
        avg_line_style="--",
        avg_line_width=1.5,
        avg_fontsize=fs - 4,  # if None, uses tick_fontsize
    ):
        """
        Create a barplot comparing model performance across skill operations.

        Args:
            dt_plot: DataFrame with columns ['dataset.skill_op', 'dataset.test_op',
                    'model_name', 'val_2op/accuracy', 'test_2op/mean_accuracy']
        """
        # Get unique values
        skill_ops = ["[SHIFT_RIGHT]", "[INVERT_POLARITY]"]
        model_names = dt_plot["model_name"].unique()
        test_ops = sorted(dt_plot["dataset.test_op"].unique())

        column_headers = [
            r"$\mathcal{C}_" + str(k) + "(S_\mathrm{new}, \Sigma_{\mathrm{train}})$",
            r"$\mathcal{C}_" + str(k) + "(S_\mathrm{new}, S_\mathrm{held\\text{-}out})$",
        ]

        metrics = [f"val_{k}op/accuracy", f"test_{k}op/mean_accuracy"]

        n_rows = len(skill_ops)
        n_cols = len(metrics)
        n_test_ops = len(test_ops)
        n_models = len(model_names)

        # Set up color palette
        if isinstance(palette, str):
            colors = sns.color_palette(palette, n_colors=n_test_ops)
        else:
            colors = palette

        # Create figure
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0.15, "wspace": 0.08},
        )
        axes = np.atleast_2d(axes)

        # Bar positioning
        group_width = n_test_ops * bar_width + (n_test_ops - 1) * bar_spacing

        for i, skill_op in enumerate(skill_ops):
            subset_skill = dt_plot[dt_plot["dataset.skill_op"] == skill_op]

            for j, metric in enumerate(metrics):
                ax = axes[i, j]

                # X positions for model groups
                x_positions = np.arange(n_models)

                # Plot bars for each test_op
                for k, test_op in enumerate(test_ops):
                    subset_test = subset_skill[subset_skill["dataset.test_op"] == test_op]

                    # Calculate bar position within group
                    offset = (k - (n_test_ops - 1) / 2) * (bar_width + bar_spacing)

                    values = []
                    for model in model_names:
                        val = subset_test[subset_test["model_name"] == model][metric].values
                        values.append(val[0] if len(val) > 0 else 0)

                    bars = ax.bar(
                        x_positions + offset,
                        values,
                        width=bar_width,
                        label=rename_op(test_op) if i == 0 and j == 0 else "",
                        color=colors[k],
                        edgecolor=edgecolor,
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=3,
                    )

                # Styling
                ax.set_xlim(-0.5, n_models - 0.5)
                ax.set_ylim(0, 1.0)

                # Grid
                if show_grid:
                    ax.yaxis.grid(True, alpha=grid_alpha, linestyle="-", zorder=0)
                    ax.set_axisbelow(True)

                # Spines
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                for spine in ["bottom", "left"]:
                    ax.spines[spine].set_linewidth(spine_linewidth)

                # X-axis labels (only on bottom row)
                if i == n_rows - 1:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(model_names, fontsize=axis_fontsize, fontweight="medium")
                else:
                    ax.set_xticks([])

                # Y-axis labels (only on left column)
                if j == 0:
                    ax.tick_params(axis="y", labelsize=tick_fontsize)
                else:
                    ax.tick_params(axis="y", labelleft=False)

                fig.supylabel("Accuracy", fontsize=axis_fontsize, fontweight="medium", x=0.04)

                # Column headers (only on top row)
                if i == 0:
                    ax.annotate(
                        column_headers[j],
                        xy=(0.5, 1.08),
                        xycoords="axes fraction",
                        ha="center",
                        va="bottom",
                        fontsize=column_header_fontsize,
                        fontweight="bold",
                    )

                # Add average lines on Sigma_test (second column)
                if show_avg and j == 1:  # j == 1 is the test column
                    # Calculate average accuracy for this skill_op across all test_ops
                    subset_for_avg = subset_skill[subset_skill["dataset.test_op"].isin(test_ops)]
                    avg_acc = subset_for_avg.groupby("model_name")[metric].mean()

                    # Draw horizontal lines for each model
                    for model_idx, model in enumerate(model_names):
                        if model in avg_acc.index:
                            y_avg = avg_acc[model]
                            x_left = model_idx - 0.35  # extend a bit beyond bars
                            x_right = model_idx + 0.35

                            # Draw the line
                            ax.plot(
                                [x_left, x_right],
                                [y_avg, y_avg],
                                color=avg_line_color,
                                linestyle=avg_line_style,
                                linewidth=avg_line_width,
                                zorder=4,
                                alpha=0.8,
                            )

                            # Add text label on the left of the line
                            ax.text(
                                x_right + 0.05,
                                y_avg,
                                f"{y_avg:.2f}",
                                ha="left",
                                va="center",
                                fontsize=avg_fontsize,
                                color=avg_line_color,
                                fontweight="medium",
                                zorder=5,
                            )

        # Add S_new header above skill labels (aligned with column headers)
        col3_x = 0.95  # x position for S_new header
        fig.text(
            col3_x,
            0.92,  # slightly below the column headers
            r"$S_{\mathrm{new}}$",
            ha="center",
            va="bottom",
            fontsize=column_header_fontsize,
            fontweight="bold",
            transform=fig.transFigure,
        )

        # Add skill operation labels on the right side of each row
        for i, skill_op in enumerate(skill_ops):
            # Format skill name nicely
            skill_name = rename_op(skill_op)

            # Add rotated text to the right of the row
            fig.text(
                col3_x,  # x position (right side)
                0.5 + (n_rows - 1 - 2 * i) / (2 * n_rows) * 0.75,  # y position (centered on row)
                f"$\\text{{{skill_name}}}$",
                rotation=0,  # rotated the other way for right side
                ha="center",
                va="center",
                fontsize=skill_label_fontsize,
                fontweight="bold",
                transform=fig.transFigure,
            )

        # Add horizontal separator line between rows (if more than one row)
        if n_rows > 1:
            for i in range(1, n_rows):
                # Get the position between row i-1 and row i
                y_sep = 1.0 - (i / n_rows) * 0.82 - 0.06  # adjust based on layout
                # Draw line from after the ylabel area to just before the skill labels
                line = plt.Line2D(
                    [
                        0.12,
                        col3_x + 0.03,
                    ],  # x: start after "Accuracy" label, end before S_new labels
                    [y_sep, y_sep],  # y: horizontal line
                    transform=fig.transFigure,
                    color="gray",
                    linewidth=0.8,
                    linestyle="-",
                    alpha=0.8,
                    zorder=10,
                )
                fig.add_artist(line)

        # Collect legend handles from first subplot
        handles, labels = axes[0, 0].get_legend_handles_labels()
        # Filter out empty labels
        handles = [h for h, l in zip(handles, labels) if l]
        labels = [l for l in labels if l]

        # Add S_test label as first entry in the legend (on same line)
        from matplotlib.patches import Rectangle

        # make invisible handle smaller
        invisible_handle = Rectangle(
            (0, 0),
            fc="w",
            fill=False,
            edgecolor="none",
            linewidth=0,
            height=0.01,
            width=0.01,
        )
        handles = [invisible_handle] + handles
        labels = [r"$S_{\mathrm{held\text{-}out}}$:    "] + [r"$\mathrm{" + l + "}$" for l in labels]

        # Add unified legend below the plot
        if legend_ncol is None:
            legend_ncol = min(len(labels), 6) + 1
        else:
            legend_ncol = legend_ncol + 1  # account for the title entry

        leg = fig.legend(
            handles,
            labels,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            frameon=True,
            fancybox=True,
            shadow=False,
            edgecolor="gray",
            facecolor="white",
            columnspacing=0.8,  # reduce spacing between columns
            handletextpad=0.5,  # reduce spacing between marker and text
        )

        # Remove individual legends from subplots
        for ax in axes.flatten():
            legend = ax.get_legend()
            if legend:
                legend.remove()

        # Adjust layout to make room for skill labels on right and legend
        plt.subplots_adjust(left=0.10, right=0.88, top=0.90, bottom=0.18)

        return fig, axes

    plot_barplot(dt_plot, k=2)
    plt.tight_layout()
    if is_save:
        # save figure
        export_dir = FIGS_DIR
        os.makedirs(export_dir, exist_ok=True)
        output_path = os.path.join(export_dir, f"neologisms_vs_baselines_k2.pdf")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    plt.show()

    plot_barplot(dt_plot, k=3)
    plt.tight_layout()
    if is_save:
        # save figure
        export_dir = FIGS_DIR
        os.makedirs(export_dir, exist_ok=True)
        output_path = os.path.join(export_dir, f"neologisms_vs_baselines_k3.pdf")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    plt.show()


def plot_icl_results():
    results_path = os.path.join(RESULTS_DIR, "skill_vs_icl_results_full.csv")
    df_results = pd.read_csv(results_path)

    # Plotting
    is_save = True
    df_plot = df_results.copy()
    df_plot = df_plot[df_plot["test_op"] != "[REVERSE]"]
    df_plot = df_plot[df_plot["seq_len"] <= 8]
    df_plot = df_plot[df_plot["k"] <= 1]
    df_plot["detail_col"] = df_plot.apply(
        lambda row: (row["test_op"] if row["method"] != "ICL" else f"ICL-{row['n_examples_per_skill']}"),
        axis=1,
    )
    df_plot.loc[df_plot["method"] == "ICL", "test_op"] = df_plot["n_examples_per_skill"]
    method2col = {
        "Skill Neologisms": COLS["blue"],
        "In-Context Learning": COLS["orange"],
    }
    from matplotlib.colors import LinearSegmentedColormap

    method2cmap = {
        "Skill Neologisms": LinearSegmentedColormap.from_list("custom_tricolor", ["black", COLS["blue"], COLS["blue_l"]], N=256),
        "In-Context Learning": LinearSegmentedColormap.from_list("custom_tricolor", ["black", COLS["orange"], COLS["orange_l"]], N=256),
    }
    # rename legend entries
    method_alias = {
        "Neologism": "Skill Neologisms",
        "ICL": "In-Context Learning",
    }
    df_plot["method"] = pd.Categorical(df_plot["method"], categories=["Neologism", "ICL"], ordered=True)
    df_plot["method"] = df_plot["method"].map(method_alias)
    lw = 2
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.lineplot(
        data=df_plot,
        x="seq_len",
        y="accuracy",
        hue="method",
        ax=ax,
        lw=lw + 3,
        marker="o",
        ci=None,
        palette=method2col,
        ls="--",
        alpha=1,
        ms=4 * lw,
    )

    for method in df_plot["method"].unique():
        from matplotlib.colors import LinearSegmentedColormap

        cmap = method2cmap[method]
        n_bars = len(df_plot[df_plot["method"] == method]["detail_col"].unique())
        colors = [cmap(0.3 + i / (n_bars - 1)) for i in range(n_bars)]

        for i, detail_col in enumerate(df_plot[df_plot["method"] == method]["detail_col"].unique()):
            df_subset = df_plot[(df_plot["method"] == method) & (df_plot["detail_col"] == detail_col)]
            sns.lineplot(
                data=df_subset,
                x="seq_len",
                y="accuracy",
                color=colors[i],  # shift to avoid full black
                ax=ax,
                lw=lw,
                alpha=0.4,
            )
        if method == "In-Context Learning":
            # annotate left most point with n_examples_per_skill
            for detail_col in df_plot[df_plot["method"] == method]["detail_col"].unique():
                df_subset = df_plot[(df_plot["method"] == method) & (df_plot["detail_col"] == detail_col)]
                x = df_subset["seq_len"].min()
                y = df_subset[df_subset["seq_len"] == x]["accuracy"].values[0]
                n_examples = detail_col.split("-")[1]
                ax.annotate(
                    f"{n_examples}",
                    xy=(0.94 * x, y),
                    xytext=(-35, -3),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.3),
                    fontsize=12,
                    color="black",
                )
            # add "# Examples" label on top of the annotations
            ax.annotate(
                "# Examples",
                xy=(1.2, 0.27),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=14,
                color="black",
            )
        elif method == "Skill Neologisms":
            # annotate the right most
            for detail_col in df_plot[df_plot["method"] == method]["detail_col"].unique():
                df_subset = df_plot[(df_plot["method"] == method) & (df_plot["detail_col"] == detail_col)]
                x = df_subset["seq_len"].max()
                y = df_subset[df_subset["seq_len"] == x]["accuracy"].values[0]
                op_name = rename_op(detail_col)
                ax.annotate(
                    f"{op_name}",
                    xy=(1.01 * x, y),
                    xytext=(15, -5),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.3),
                    fontsize=13,
                    color="black",
                )
            ax.annotate(
                r"$S_\mathrm{held\text{-}out}$",
                xy=(df_plot["seq_len"].max(), 0.58),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=18,
                color="black",
            )
    plt.xlim(1.01, 8.9)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_xlabel("Sequence Length (↑ harder)", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    format_fig(plt.gcf(), axis_fontsize=16, tick_fontsize=14)
    ax.legend(
        title="",
        fontsize=12,
        title_fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
    )
    if is_save:
        export_dir = FIGS_DIR
        os.makedirs(export_dir, exist_ok=True)
        output_path = os.path.join(export_dir, f"plot_icl_results_detailed.pdf")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    plt.show()

    # same without the individual lines
    fig, ax = plt.subplots(figsize=(8, 2.5))
    sns.lineplot(
        data=df_plot,
        x="seq_len",
        y="accuracy",
        hue="method",
        ax=ax,
        lw=lw + 3,
        marker="o",
        ci="sd",
        palette=method2col,
        ls="--",
        alpha=1,
        ms=4 * lw,
    )
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_xlabel("Sequence Length (↑ harder)", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    format_fig(plt.gcf(), axis_fontsize=16, tick_fontsize=14)
    ax.legend(
        title="",
        fontsize=12,
        title_fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=2,
    )
    if is_save:
        export_dir = FIGS_DIR
        os.makedirs(export_dir, exist_ok=True)
        output_path = os.path.join(export_dir, f"icl_results.pdf")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate toy-experiment figures/tables from WandB logs and cached CSVs.")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "p2_baselines",
            "p3_skill_vs_icl",
        ],
        help="Which analysis targets to run.",
    )
    parser.add_argument(
        "--pretrain-eval-csv",
        default=None,
        help="Path to eval_results_1op.csv for base pretrain accuracy figure. Used by target=pretrain_skill_acc.",
    )
    args = parser.parse_args()

    targets = args.targets
    if "all" in targets:
        targets = [
            "p2_baselines",
            "p3_skill_vs_icl",
        ]

    if "p2_baselines" in targets:
        plot_neologisms_vs_baselines(tags=["lora", "prompt-tuning", "skill-tokens"], reload_cache=True)
    if "p3_skill_vs_icl" in targets:
        plot_icl_results()


if __name__ == "__main__":
    main()
