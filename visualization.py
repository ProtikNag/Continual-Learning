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
    num_tasks = len(naive_learning_performance)
    fig, axes = plt.subplots(1, num_tasks, figsize=(20, 5), sharey=True)
    width = 0.35
    x_labels = [f'Task {i + 1}' for i in range(num_tasks)]

    for i in range(num_tasks):
        x = np.arange(num_tasks)

        naive_values = [naive_learning_performance[i][j] for j in range(num_tasks)]
        mer_values = [mer_performance[i][j] for j in range(num_tasks)]

        axes[i].bar(x - width / 2, naive_values, width, label='Naive Learning', color='orange')
        axes[i].bar(x + width / 2, mer_values, width, label='MER', color='blue')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(x_labels)
        axes[i].set_title(f'Performance After Training on Task {i + 1}')
        axes[i].set_xlabel('Tasks')

        if i == 0:
            axes[i].set_ylabel('Accuracy')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    plt.suptitle('Naive Learning vs MER Performance on All Tasks', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('./Figures/' + title + '.pdf')
    plt.show()


def plot_impact_of_parameters(performance_results):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    scatter = ax[0].scatter(performance_results[:, 0], performance_results[:, 1], c=performance_results[:, 2],
                            cmap='viridis')
    ax[0].set_title('MER Performance (Accuracy) vs Beta, Gamma')
    ax[0].set_xlabel('Beta')
    ax[0].set_ylabel('Gamma')
    fig.colorbar(scatter, ax=ax[0])
    plt.tight_layout()
    plt.savefig('./Figures/Impact_of_Parameters.pdf')
    plt.show()