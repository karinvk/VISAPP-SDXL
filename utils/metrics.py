#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, precision_score, recall_score
import pandas as pd

def get_metrics(labels, predictions):
    metrics = {}
    
    ap_score=average_precision_score(labels, predictions)
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    ap_score_by_auc = auc(recall, precision)

    binary_predictions = np.where(predictions > 0.5, 1, 0)
    recall_score_macro=recall_score(labels, binary_predictions, average='macro',zero_division=0)
    precision_score_macro=precision_score(labels, binary_predictions, average='macro',zero_division=0)

    metrics['ap_score']=ap_score
    metrics['ap_score_auc']=ap_score_by_auc
    metrics['precision_score']=precision_score_macro
    metrics['recall_score']=recall_score_macro

    
    return metrics
