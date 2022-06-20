import numpy as np
from sklearn import metrics


def estimator(biased, prediction):
    # define AUC using sklearn.metrics
    fpr, tpr, thresholds = metrics.roc_curve(biased, prediction[:, 1])
    auc = metrics.auc(fpr, tpr)

    # the threshold selection of MAX (sensitivity + specificity)
    max_tpr_fpr = np.array(tpr + (1 - fpr))
    threshold = thresholds[np.argmax(max_tpr_fpr)]

    # create the biased predictions
    y_prediction = np.array([1 if i[1] >= threshold else 0 for i in prediction])

    # define true negative, false positive, false negative, true positive using biased predictions
    tn, fp, fn, tp = metrics.confusion_matrix(biased, y_prediction).ravel()

    # calculate Cohen's Kappa and TSS metrics
    kappa = (2 * (tp * tn - fn * fp)) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))
    spec = tp / (tp + fn)  # also equals to 1 - fpr
    sens = tn / (tn + fp)  # also equals to tpr
    tss = sens + spec - 1

    return auc, tss, kappa, threshold
