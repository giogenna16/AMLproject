import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses)
    plt.show()

    plt.figure()
    plt.plot(range(len(val_losses)), val_losses)
    plt.show()
