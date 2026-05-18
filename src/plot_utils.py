COLS = {
    "blue": "#3E92CC",
    "orange": "#FCB173",
    "green": "#60B08C",
    "red": "#D8315B",
    "almond": "#E0C1B3",
    "purple": "#685F74",
    "blue_l": "#8EBFE1",
    "orange_l": "#FDC79B",
    "green_l": "#9FD0BA",
    "red_l": "#EFA9BB",
    "almond_l": "#F3E7E2",
    "purple_l": "#ADA5B6",
}


AXIS_FONTSIZE = 18
TICK_FONTSIZE = 16


def format_fig(fig, axis_fontsize=AXIS_FONTSIZE, tick_fontsize=TICK_FONTSIZE):
    for ax in fig.axes:
        ax.xaxis.label.set_size(axis_fontsize)
        ax.yaxis.label.set_size(axis_fontsize)
        ax.title.set_size(axis_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=tick_fontsize)
    return fig
