from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from scipy.special import softmax
import seaborn as sns


def plot_multiclass_pr_curves(y_pred, y_true, label_d=None, from_logits=False, title=None, save=None):
    if label_d is None:
        num_classes_rng = range(y_pred.shape[-1])
        label_d = dict(zip(num_classes_rng, map(str, num_classes_rng)))

    if from_logits:
        y_pred = softmax(y_pred, axis=-1)

    plt.figure(figsize=(10, 7))
    aps = []
    for idx, label in label_d.items():
        y_pred_single_class = y_pred[:, idx]

        y_true_single_class = np.zeros_like(y_true)
        y_true_single_class[y_true == idx] = 1

        precision, recall, _ = precision_recall_curve(y_true_single_class, y_pred_single_class)
        ap = average_precision_score(y_true_single_class, y_pred_single_class)
        plt.step(recall, precision, where='post', label=f'{label}: AP={ap:.3f}')
        aps.append(ap)

    plt.ylim(-.05, 1.05)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc='best')
    if title is None:
        plt.title(f'Precision Recall Curves, mAP={mean(aps):.3f}')
    else:
        plt.title(f'{title} Precision Recall Curves, mAP={mean(aps):.3f}')
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_confusion_matrix(y_pred, y_true, label_d=None, title=None, save=None):
    if label_d is None:
        num_classes_rng = range(y_pred.shape[-1])
        label_d = dict(zip(num_classes_rng, map(str, num_classes_rng)))

    cm = confusion_matrix(y_true, np.argmax(y_pred, axis=-1))
    df_cm = pd.DataFrame(cm, index=[label_d[i] for i in range(len(label_d))],
                         columns=[label_d[i] for i in range(len(label_d))])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    if title is None:
        plt.title('Confusion Matrix')
    else:
        plt.title(f'{title} Confusion Matrix')
    if save is not None:
        plt.savefig(save + '.png')
