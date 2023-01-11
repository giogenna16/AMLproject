import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, val_accuracies):
    
    plt.figure()
    plt.title("train_losses")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.grid()
    x_axis_list = [x * 100 for x in list(range(len(train_losses)))]
    plt.plot(x_axis_list, train_losses)
    plt.savefig("/content/drive/MyDrive/AML_project/plot_train_loss.png")
    plt.show()

    plt.figure()
    plt.title("validation_losses")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.grid()
    x_axis_list = [x * 100 for x in list(range(len(val_losses)))]
    plt.plot(x_axis_list, val_losses)
    plt.savefig("/content/drive/MyDrive/AML_project/plot_validation_loss.png")
    plt.show()

    plt.figure()
    plt.title("validation_accuracies")
    plt.xlabel("iteration")
    plt.ylabel("accuracy [%]")
    plt.grid()
    x_axis_list = [x * 100 for x in list(range(len(val_accuracies)))]
    plt.plot(x_axis_list, val_accuracies)
    plt.savefig("/content/drive/MyDrive/AML_project/plot_validation_accuracy.png")
    plt.show()
