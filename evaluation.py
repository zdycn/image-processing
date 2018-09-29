import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_similarity_score
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    # plt.plot(epochs, acc, 'b-', label='Training acc')
    # plt.plot(epochs, val_acc, 'r', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.figure()

    plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
    plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    # plt.plot(epochs, loss, 'b-', label='Training loss')
    # plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    # plt.title('Training and validation loss')

    plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
    plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points








def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix',
                          cmap=plt.cm.Blues, is_save=False, save_path=""):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    if is_save:
        if (save_path == ""):
            plt.savefig('confusion_matrix', dpi=200)
        else:
            plt.savefig(os.path.join(save_path, 'confusion_matrix'), dpi=200)

def evaluate_indicator(ground_truth, predicted_classes):
    """
        计算学习模型的各项评价指标
    :param ground_truth:
    :param predicted_classes:
    :return:
    """
    confusion = confusion_matrix(ground_truth, predicted_classes)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    target_names = ['0', '1']

    plot_confusion_matrix(confusion, target_names)

    print(classification_report(ground_truth, predicted_classes, target_names=target_names))

    acc = round(accuracy_score(ground_truth, predicted_classes), 3)
    precision = round(precision_score(ground_truth, predicted_classes), 3)
    recall = round(recall_score(ground_truth, predicted_classes), 3)
    specificity = round(TN / float(TN+FP), 3)
    sensitivity = round(TP / float(TP+FN), 3)
    f1 = round(f1_score(ground_truth, predicted_classes), 3)
    auc = round(roc_auc_score(ground_truth, predicted_classes), 3)

    # kaggle竞赛采用的指标，score>0.8意味着好的分类；0或更低意味着不好（实际是随机标签）
    kappa_score = round(cohen_kappa_score(ground_truth, predicted_classes), 3)

    ind = "acc\tprecision\trecall\tspecificity\tsensitivity\tF-Score\tauc\tkappa-score\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
    print(ind.format(acc, precision, recall, specificity, sensitivity, f1, auc, kappa_score))

    # Jaccard similarity index
    jaccard_index = jaccard_similarity_score(ground_truth, predicted_classes, normalize=True)
    print("\nJaccard similarity score: " + str(jaccard_index))

    # Save the results
    file_perf = open('performances.txt', 'w')
    file_perf.write("\nACCURACY: " + str(acc)
                    + "\nPRECISION: " + str(precision)
                    + "\nRECALL: " + str(recall)
                    + "\nSPECIFICITY: " + str(specificity)
                    + "\nSENSITIVITY: " + str(sensitivity)
                    + "\nF1 score (F-measure): " + str(f1)
                    + "Area under the ROC curve: " + str(auc)
                    + "\nKappa Score: " + str(kappa_score)
                    + "\nJaccard similarity score: " + str(jaccard_index)
                    + "\n\nConfusion matrix:" + str(confusion))
    file_perf.close()

def plot_roc(y_true, y_scores):
    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    roc_curve_fig = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("ROC.png")


def plot_prec_rec_curve(y_true, y_scores):
    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig("Precision_recall.png")