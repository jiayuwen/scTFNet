import numpy as np
import csv
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    cohen_kappa_score, auc, roc_curve, matthews_corrcoef, precision_recall_curve, roc_auc_score


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f * mask)
    return (2. * intersection + smooth) / (K.sum(y_true_f * mask) + K.sum(y_pred_f * mask) + smooth)


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def f1_metrics(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*p*r / (p + r + K.epsilon())


def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

# def evaluation(model, input_seq, input_label, tf_idx=0):
#     pre_label = np.round(model.predict(input_seq))[:, :, tf_idx]
#     true_label = np.array(input_label)[:, :, tf_idx, :]
#     acc = []
#     f1 = []
#     a = []
#     mcc = []
#     p = []
#     r = []

#     for i in range(len(true_label)):
#         acc.append(accuracy_score(true_label[i], pre_label[i]))
#         if len(set(true_label[i][:, 0])) == 2 and len(set(pre_label[i])) == 2:
#             f1.append(f1_score(true_label[i], pre_label[i]))
#             mcc.append(matthews_corrcoef(true_label[i], pre_label[i]))
#             a.append(roc_auc_score(true_label[i], pre_label[i]))
#             p.append(precision_score(true_label[i], pre_label[i]))
#             r.append(recall_score(true_label[i], pre_label[i]))

#     result_dic = {
#         'Accuracy': np.average(np.array(acc)),
#         'F-1': np.average(np.array(f1)),
#         'AUC': np.average(np.array(a)),
#         'MCC': np.average(np.array(mcc)),
#         'Precision': np.average(np.array(p)),
#         'Recall': np.average(np.array(r))}

#     return result_dic


def save_result(result_dic, path=''):
    with open(path + "evaluation_metrics.csv", "w") as file:
        writer = csv.DictWriter(file, result_dic.keys())
        writer.writeheader()
        writer.writerow(result_dic)

