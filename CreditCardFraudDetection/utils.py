import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import config


class ConfusionMatrixPlotter:
    def __init__(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        self.cm = cm
        self.classes = classes
        self.normalize = normalize
        self.title = title
        self.cmap = cmap

    def plot(self, save_plot = False, model_type= None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if self.normalize:
            self.cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(self.cm)

        plt.imshow(self.cm, interpolation='nearest', cmap=self.cmap)
        plt.title(self.title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        fmt = '.2f' if self.normalize else 'd'
        thresh = self.cm.max() / 2.
        for i, j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):
            plt.text(j, i, format(self.cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if self.cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        if save_plot == True:
            if not os.path.exists(config.DataConfig.GRAPH_DIR):
                os.makedirs(config.DataConfig.GRAPH_DIR)
            save_path = os.path.join(config.DataConfig.GRAPH_DIR,
                                     f'{model_type}_confusion_matrix.png')
            plt.savefig(save_path)
        else:
            plt.show()
