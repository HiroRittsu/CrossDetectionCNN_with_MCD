import numpy as np
from pytorch_lightning.metrics.functional import iou
from sklearn.metrics import recall_score

SMOOTH = 1e-15


def categorical_accuracy(pred_y, true_y):
    pred_y = pred_y.max(dim=1)[1]
    acc = pred_y.eq(true_y.data).float().mean().item()
    return acc


def binary_accuracy(pred_y, true_y):
    assert true_y.size() == pred_y.size(), "pred_y:{}, true_y,{}".format(pred_y.shape, true_y.shape)
    return (pred_y > 0.5).eq(true_y.data).float().mean().item()


def mean_accuracy(pred_y, true_y, mode='mean'):
    """
    :param pred_y:
    :param true_y:
    :param mode:
    :return:
    """
    score_list = []
    pred_y = pred_y.max(dim=1)[1].cpu().numpy()
    true_y = true_y.cpu().numpy()
    for pred, true in zip(pred_y, true_y):
        m_acc_scores = []
        for label_id in np.unique(true):
            p_bool_tensor = np.where(pred == label_id, True, False)
            t_bool_tensor = np.where(true == label_id, True, False)
            common_tensor = (p_bool_tensor * t_bool_tensor)
            # mean_acc
            m_acc = np.sum(common_tensor) / np.sum(t_bool_tensor)
            m_acc_scores.append(m_acc)
        score_list.extend(m_acc_scores)
    if len(score_list) == 0:
        score_list.append(0)
    return np.nanmean(score_list) if mode == "mean" else score_list


def iou_metrics(pred_y, true_y, mode='elementwise_mean', ignore_id=None):
    """
    :param ignore_id:
    :param pred_y:
    :param true_y:
    :param mode:
    :return:
    """
    pred_y = pred_y.max(dim=1)[1].view(-1).cpu()
    true_y = true_y.view(-1).cpu()
    return iou(pred_y, true_y, reduction=mode, ignore_index=ignore_id).numpy()


def recall_metrics(pred_y, true_y, mode='mean', ignore_id: list = None):
    """
    :param ignore_id:
    :param pred_y:
    :param true_y:
    :param mode:
    :return:
    """
    score_list = []
    num_class = pred_y.shape[1]
    pred_y = pred_y.max(dim=1)[1].view(-1).cpu()
    true_y = true_y.view(-1).cpu()
    for label_id in range(num_class):
        if ignore_id is not None and label_id in ignore_id:
            continue
        bin_pred_y = pred_y.eq(label_id).int().numpy()
        bin_true_y = true_y.eq(label_id).int().numpy()
        score = recall_score(bin_true_y, bin_pred_y)
        score_list.append(score if score > SMOOTH else 0)
    return np.nanmean(score_list) if mode == "mean" else score_list


def dice_metrics(pred_y, true_y, mode='mean', ignore_id: list = None):
    """
    :param ignore_id:
    :param mode:
    :param pred_y:
    :param true_y:
    :return:
    """
    score_list = []
    num_class = pred_y.shape[1]
    pred_y = pred_y.max(dim=1)[1].view(-1).cpu()
    true_y = true_y.view(-1).cpu()
    for label_id in range(num_class):
        if ignore_id is not None and label_id in ignore_id:
            continue
        bin_pred_y = pred_y.eq(label_id).int()
        bin_true_y = true_y.eq(label_id).int()
        if bin_true_y.sum() == 0 and mode == "mean":
            continue
        bin_common = bin_pred_y * bin_true_y
        score = ((2 * bin_common.sum()) / (bin_pred_y.sum() + bin_true_y.sum() + SMOOTH)).item()
        score_list.append(score if score > SMOOTH else 0)
    return np.nanmean(score_list) if mode == "mean" else score_list
