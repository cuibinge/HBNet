import os
import cv2
import numpy as np


def calculate_iou(mask_true, mask_pred):
    """
    计算交并比 (IoU)
    """
    # 计算交集与并集
    intersection = np.logical_and(mask_true, mask_pred)
    union = np.logical_or(mask_true, mask_pred)
    # 防止除零错误
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
    return iou

def calculate_metrics(mask_true, mask_pred):
    # 计算真阳性(TP)、假阳性(FP)、真阴性(TN)、假阴性(FN)
    TP = np.sum(np.logical_and(mask_true, mask_pred))
    FP = np.sum(np.logical_and(np.logical_not(mask_true), mask_pred))
    TN = np.sum(np.logical_and(np.logical_not(mask_true), np.logical_not(mask_pred)))
    FN = np.sum(np.logical_and(mask_true, np.logical_not(mask_pred)))

    # 计算精确度、准确度、召回率和 F1 score
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    iou = calculate_iou(mask_true, mask_pred)

    return precision, accuracy, recall, iou, f1_score

def batch_metrics(true_dir, pred_dir):
    precisions = []
    accuracies = []
    recalls = []
    ious = []
    f1_scores = []

    for filename in os.listdir(true_dir):
        if filename.endswith('.png'):
            mask_true = cv2.imread(os.path.join(true_dir, filename), 0)
            mask_pred = cv2.imread(os.path.join(pred_dir, filename), 0)

            # 调整掩码大小到相同尺寸
            #mask_true = cv2.resize(mask_true, (512, 512), interpolation=cv2.INTER_LINEAR)
            #mask_pred = cv2.resize(mask_pred, (512, 512), interpolation=cv2.INTER_LINEAR)

            precision, accuracy, recall, iou, f1_score = calculate_metrics(mask_true, mask_pred)
            precisions.append(precision)
            accuracies.append(accuracy)
            recalls.append(recall)
            ious.append(iou)
            f1_scores.append(f1_score)

    mean_precision = np.mean(precisions)
    mean_accuracy = np.mean(accuracies)
    mean_recall = np.mean(recalls)
    mean_iou = np.mean(ious)
    mean_f1_score = np.mean(f1_scores)

    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean F1 Score: {mean_f1_score:.4f}")

    return mean_precision, mean_accuracy, mean_recall, mean_iou, mean_f1_score

# 示例用法
true_dir = r'D:\github\liumengting\HBNet\data\test\mask'
pred_dir = r'D:\github\liumengting\test_out\hb_ori_no_modify'
mean_precision, mean_accuracy, mean_recall, mean_iou, mean_f1_score = batch_metrics(true_dir, pred_dir)
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Mean F1 Score: {mean_f1_score:.4f}")
