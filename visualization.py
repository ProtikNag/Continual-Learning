import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(losses, title='Training Loss Curve', xlable='Epoch', ylabel='Loss', case="All Data Together"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', color='b')
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig('./Figures/' + case + '.pdf')
    plt.show()


def plot_performance_histograms(naive_learning_performance, mer_performance, title='Performance Histograms'):
    # Number of tasks (assuming keys in both dictionaries are the same)
    num_tasks = len(naive_learning_performance)

    # Create subplots for each "After Training on Task i"
    fig, axes = plt.subplots(1, num_tasks, figsize=(20, 5), sharey=True)

    width = 0.35  # width of the bars
    x_labels = [f'Task {i + 1}' for i in range(num_tasks)]

    for i in range(num_tasks):
        # Initialize an array for bar positions (4 tasks → 4*2 = 8 bars in total)
        x = np.arange(num_tasks)

        # Retrieve performance data for each task after training on i-th task
        naive_values = [naive_learning_performance[i][j] for j in range(num_tasks)]
        mer_values = [mer_performance[i][j] for j in range(num_tasks)]

        # Plot Naive Learning and MER performance side by side
        axes[i].bar(x - width / 2, naive_values, width, label='Naive Learning', color='orange')
        axes[i].bar(x + width / 2, mer_values, width, label='MER', color='blue')

        # Set x-axis labels
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(x_labels)
        axes[i].set_title(f'Performance After Training on Task {i + 1}')
        axes[i].set_xlabel('Tasks')

        if i == 0:
            axes[i].set_ylabel('Accuracy')

    # Add a legend to the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)

    # Set the main title for the entire figure
    plt.suptitle('Naive Learning vs MER Performance on All Tasks', fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


# naive_learning_performance = {
#     0: [0.9964788732394366, 0.0, 0.0, 0.0],
#     1: [0.3137472283813747, 0.6812638580931264, 0.005543237250554324, 0.009423503325942351],
#     2: [0.2538116591928251, 0.5511210762331838, 0.19551569506726457, 0.008071748878923767],
#     3: [0.20901033973412111, 0.4538404726735598, 0.16100443131462333, 0.11669128508124077]
# }
#
# mer_performance = {
#     0: [0.9929577464788732, 0.9947183098591549, 0.9964788732394366, 0.9964788732394366],
#     1: [0.31263858093126384, 0.9983370288248337, 0.9988913525498891, 0.9988913525498891],
#     2: [0.25291479820627805, 0.8076233183856503, 0.9991031390134529, 0.9991031390134529],
#     3: [0.20827178729689808, 0.6650664697193501, 0.8227474150664698, 0.9981536189069424]
# }
#
# plot_performance_histograms(naive_learning_performance, mer_performance)
