import matplotlib.pyplot as plt
import os


def plot_pr_auc(history, path=''):
    # draw PR_AUC between training and validation data

    fig = plt.figure(0, figsize=(8, 6))
    plt.plot(history.history['pr_auc'])
    plt.plot(history.history['val_pr_auc'])
    plt.title('Model PR_AUC')
    plt.ylabel('pr_auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    if not os.path.isdir(path):
        return "Can not save pr_auc, since the path is invalid"
    else:
        plt.savefig(path + f'PR_AUC_training_history.png')
        return "PR_AUC training history save to:" + path
    plt.close(fig)