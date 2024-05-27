import catboost
import os
import pandas as pd
import numpy as np


import sys
from pathlib import Path
from sklearn.isotonic import spearmanr
from sklearn.metrics import f1_score

from my_func_tools import print_classify_result


# 本文件目录
real_dir = Path(__file__).resolve().parent
# 需要转str才行
sys.path.append(str(real_dir.parent))


current_dir = os.path.split(os.path.realpath(__file__))[0]
 
def get_importance_catboost(bst, importance_file):
    # 按importance_gain排序
    df_important = pd.DataFrame(
        {"feature_name": bst.feature_names_, "importance": bst.feature_importances_}
    )
    df_important = df_important.sort_values(by=["importance"], ascending=False)
    df_important.to_csv(importance_file, index=False)


# 提供超参数
def train_a_2_label_catboostoostClassifier(
    train_input,
    train_label,
    val_input,
    val_label,
    current_cat_feature,
    init_model=None,
    # 分类器参数
    **kwargs,
):
    DEFAULT_PARAMS = dict(
        iterations=300,
        eval_metric="TotalF1:average=Macro",
        # eval_metric="TotalF1:average=Micro",
        # eval_metric="TotalF1:average=Weighted",
        learning_rate=0.07,
        depth=6,
        subsample=None,
        colsample_bylevel=None,
        min_data_in_leaf=None,
        verbose=20,
        early_stopping_rounds=60,
        loss_function="Logloss",
    )

    # 合并默认参数和传入参数  后面的kwargs覆盖前面的DEFAULT_PARAMS相同key
    params = {**DEFAULT_PARAMS, **kwargs}

    # 训练
    bst = catboost.CatBoostClassifier(**params)

    # 根据有没有val来设置不同的eval_set
    if val_input is None:
        eval_set = catboost.Pool(
            data=train_input, label=train_label, cat_features=current_cat_feature
        )

    else:
        eval_set = catboost.Pool(
            data=val_input, label=val_label, cat_features=current_cat_feature
        )

    # 直接训练模型和进行预测
    bst.fit(
        X=train_input,
        y=train_label,
        cat_features=current_cat_feature,
        # use_best_model=False
        use_best_model=True,
        init_model=init_model,
        eval_set=eval_set,
    )
    get_importance_catboost(bst, "./z_catboost_importance.csv")

    return bst


def test_a_2_label_catboostoostClassifier(
    bst: catboost.CatBoostClassifier,
    test_input,
    # 可以不传
    test_label_list=None,
    # 预测的树的数量 best就提取bst的best 其他就直接传值就行 None或者int
    # ntree_end=None,
    ntree_end="best",
    output=True,
    threshold=0.5,
):
    if ntree_end == "best":
        # catboost iteration从0开始计数 所以涉及到总数的时候往往是best_iteration_+1
        # 而ntree_end是开区间 所以也是best_iteration+1
        # tree_count_就是iteration数量 如果用了use_best_model=True 那么它就是best_iteration_+1
        ntree_end = bst.best_iteration_ + 1

    # 二分类中标签为1的score
    pred_score_list = bst.predict_proba(
        X=test_input,
        ntree_end=ntree_end,
    )[:, 1]
    pred_score_list = pred_score_list

    # 标签
    pred_label_list = (pred_score_list > threshold).astype(int).tolist()

    if test_label_list is None:
        classify_report_dict = None
    else:
        classify_report_dict = print_classify_result(
            test_label_prediction_list=pred_label_list,
            test_label_list=test_label_list,
            output=output,
            test_score_list=pred_score_list,
        )

    return pred_score_list, pred_label_list, classify_report_dict
