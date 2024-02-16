"""_summary_
    Metrics include APCER, BPCER, AUC, Accuracy, Precision, Recall, EER, HTER
    FAS
"""
import torch

def num_tp(target, pred):
    return torch.sum(torch.logical_and(pred == 1, target == 1))


def num_fp(target, pred):
    return torch.sum(torch.logical_and(pred == 1, target == 0))


def num_fn(target, pred):
    return torch.sum(torch.logical_and(pred == 0, target == 1))


def num_tn(target, pred):
    return torch.sum(torch.logical_and(pred == 0, target == 0))


def apcer(fp, tn):
    return fp / (fp + tn)


def bpcer(fn, tp):
    return fn / (fn + tp)


def acer(tp, tn, fp, fn):
    return (apcer(fp, tn) + bpcer(fn, tp)) / 2


def fpr(fp, tn):
    return fp / (fp + tn)


def tpr(tp, fn):
    return tp / (tp + fn)