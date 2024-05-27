from lightgbm import (
    LGBMRegressor,
    LGBMClassifier,
    Booster,
    early_stopping,
    log_evaluation,
)
import os
import lightgbm
import pandas as pd
import numpy as np


import sys
from pathlib import Path
from sklearn.metrics import f1_score

from my_func_tools import print_classify_result

# 本文件目录
real_dir = Path(__file__).resolve().parent
# 需要转str才行
sys.path.append(str(real_dir.parent))


current_dir = os.path.split(os.path.realpath(__file__))[0]


def get_importance_lightgbm(bst, importance_file):
    # 按importance_gain排序
    df_important = (
        pd.DataFrame(
            {
                "feature_name": bst.feature_name(),
                "importance_gain": bst.feature_importance(importance_type="gain"),
                "importance_split": bst.feature_importance(importance_type="split"),
            }
        )
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )

    # 存储
    df_important.to_csv(importance_file, index=False)


# 定义加权 F1 分数的自定义评估函数
def weighted_f1_score(preds, train_data):
    labels = train_data.get_label()
    preds_binary = (preds > 0.5).astype(int)  # 转换为二分类预测
    f1 = f1_score(labels, preds_binary, average="weighted")
    return "weighted_f1_score", f1, True


def marco_f1_score(preds, train_data):
    labels = train_data.get_label()
    preds_binary = (preds > 0.5).astype(int)  # 转换为二分类预测
    f1 = f1_score(labels, preds_binary, average="macro")
    return "marco_f1_score", f1, True


def train_a_2_label_lgbmclassifier(
    train_input, train_label, val_input, val_label, current_cat_feature, **kwargs
):

    DEFAULT_PARAMS = {
        "boosting_type": "gbdt",  # 提升类型为梯度提升决策树
        "objective": "binary",  # 任务类型为二分类
        "verbose": -1,
        "n_jobs": -1,
        "device": "cpu",
        "random_state": 1,
        # "is_unbalance": True,  # 二分类train样本不均衡 所以要设置这个
        # "metric": "binary_logloss",
        "metric": "None",
        # "num_leaves": 16,
        # "max_depth": 4,
        # "min_data_in_leaf": 20,
        "learning_rate": 0.03,
        # 添加的参数
        "iterations": 300,
        "feval": marco_f1_score,
        "early_stopping_rounds": 60,
        "log_evaluation": False,
    }

    # 合并默认参数和传入参数  后面的kwargs覆盖前面的DEFAULT_PARAMS相同key
    params = {**DEFAULT_PARAMS, **kwargs}
    iterations = params.pop("iterations")
    feval = params.pop("feval")
    early_stopping_rounds = params.pop("early_stopping_rounds")
    log_evaluation = params.pop("log_evaluation")

    # 回调函数
    callback_funcs = []
    if log_evaluation:
        callback_funcs.append(lightgbm.log_evaluation(period=20, show_stdv=True))

    train_dataset = lightgbm.Dataset(train_input, train_label)
    if val_input is None:
        val_kwargs = {"valid_sets": [train_dataset], "valid_names": ["train"]}
    else:
        val_dataset = lightgbm.Dataset(val_input, val_label)
        val_kwargs = {"valid_sets": [val_dataset], "valid_names": ["val"]}
        callback_funcs += [
            lightgbm.early_stopping(early_stopping_rounds, first_metric_only=False),
        ]

    bst = lightgbm.train(
        params=params,
        train_set=train_dataset,
        num_boost_round=iterations,  # 默认100,别名=num_iteration,
        feval=feval,  # 评估函数
        categorical_feature=current_cat_feature,
        callbacks=callback_funcs,
        **val_kwargs
    )

    # 重要性
    get_importance_lightgbm(bst, "./z_lgbm_importance.csv")

    return bst


def test_a_2_label_lightgbmClassifier(
    bst: lightgbm.Booster,
    test_input,
    # 可以不传
    test_label_list=None,
    # 预测的树的数量 best就提取bst的best 其他就直接传值就行 None或者int
    # ntree_end=None,
    num_iteration="best",
    output=True,
):
    if num_iteration == "best":
        num_iteration = bst.best_iteration

    # 二分类predcit返回的是正类即1的概率 所以一般都是用0.5判断
    pred_score_list = bst.predict(test_input, num_iteration=bst.best_iteration)
    pred_label_list = (pred_score_list > 0.5).astype(int).tolist()

    if test_label_list is None:
        classify_report_dict = None
    else:
        classify_report_dict = print_classify_result(
            test_label_prediction_list=pred_label_list,
            test_label_list=test_label_list,
            output=output,
            test_score_list=pred_score_list
        )

    return pred_score_list, pred_label_list, classify_report_dict
