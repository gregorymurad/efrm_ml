import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# Generates a customized graph
def generateGraph(lines_list,
                  lines_names,
                  title='',
                  figsize=(9, 6),
                  xlabel='Sample number',
                  ylabel='Measurements',
                  grid=True,
                  legend=None,
                  font_size=13,
                  font_family='serif', **kwargs):
    if legend == None:
        legend = len(lines_list) > 0
    # Set fonts
    matplotlib.rcParams.update({'font.size': font_size, 'font.family': font_family})
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    # Plot
    for line, name in zip(lines_list, lines_names):
        ax.plot(line, label=name)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
    if legend:
        ax.legend()
    if grid:
        ax.grid()
    ax.set_title(title)
    return fig, ax


# Buils a corelation matrix for the elements in the pandas dataframe

def generateCorrHeatmap(dataframe, method, labels, heatMapKwargs={}, title='',
                        figsize=(3, 3),
                        xlabel='Sample number',
                        ylabel='Measurements',
                        grid=True,
                        legend=None,
                        font_size=13,
                        font_family='serif'):
    matplotlib.rcParams.update({'font.size': font_size, 'font.family': font_family})
    dataframe = dataframe.drop(['Time hh:mm:ss'], axis=1)
    dataframe = dataframe.rename({col: lab for col, lab in zip(list(dataframe.columns), labels)}, axis=1)
    corr = dataframe.corr(method=method)
    graph = sns.heatmap(corr, **heatMapKwargs)
    graph.set_xticklabels(graph.get_xticklabels(), rotation=20, fontsize=2 * font_size / 3)
    graph.set_yticklabels(graph.get_yticklabels(), rotation=0, fontsize=2 * font_size / 3)
    return graph