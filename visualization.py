import matplotlib.pyplot as plt


def plot_loss_curve(losses, title='Training Loss Curve', xlable='Epoch', ylabel='Loss', case="All Data Together"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', color='b')
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig('./Figures/' + case + '.pdf')
    plt.show()

