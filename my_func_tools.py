from collections import defaultdict


from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    roc_auc_score,
)


def format_dict_numbers(input_dict, decimal_places=4):
    """
    格式化字典中的数字为指定小数位数

    Args:
        input_dict (dict): 输入字典
        decimal_places (int): 保留的小数位数

    Returns:
        dict: 格式化后的字典
    """
    formatted_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            formatted_value = {
                k: round(v, decimal_places) if isinstance(v, float) else v
                for k, v in value.items()
            }
        else:
            formatted_value = (
                round(value, decimal_places) if isinstance(value, float) else value
            )
        formatted_dict[key] = formatted_value
    return formatted_dict


def print_classify_result(
    test_label_prediction_list, test_label_list, output=True, test_score_list=None
):

    # 传入工厂函数来设置默认值
    # 计数类型的error_dict
    # error_dict = defaultdict(lambda: 0)
    error_dict = defaultdict(lambda: [])
    for idx, (i, j) in enumerate(zip(test_label_list, test_label_prediction_list)):
        if i != j:
            # error_dict[(i, j)] += 1
            # 记录错误idx类型的error_dict
            error_dict[(i, j)].append(idx)

    # 获得dict类型的分类结果
    classify_report_dict = classification_report(
        y_true=test_label_list,
        y_pred=test_label_prediction_list,
        zero_division=0,
        output_dict=True,
    )
    # 计算 Micro F1
    micro_f1 = f1_score(
        y_true=test_label_list, y_pred=test_label_prediction_list, average="micro"
    )
    # 计算 Marco F1
    macro_f1 = f1_score(
        y_true=test_label_list, y_pred=test_label_prediction_list, average="macro"
    )
    weighted_f1 = f1_score(
        y_true=test_label_list, y_pred=test_label_prediction_list, average="weighted"
    )
    if test_score_list is not None:
        auc = roc_auc_score(y_true=test_label_list, y_score=test_score_list)
    else:
        auc = None

    # -1的时候除0 返回0
    if output:
        classify_report = classification_report(
            y_true=test_label_list,
            y_pred=test_label_prediction_list,
            zero_division=0,
            digits=4,
            # output_dict=True,
        )
        print(f"error_dict{error_dict}", flush=True)
        print(
            classify_report,
            flush=True,
        )
        # 输出
        print(
            f"测试集accuracy{classify_report_dict['accuracy']} micro_f1 {micro_f1} macro_f1 {macro_f1}",
            flush=True,
        )

    # 添加f1记录
    classify_report_dict["micro_f1"] = micro_f1
    classify_report_dict["macro_f1"] = macro_f1
    classify_report_dict["weighted_f1"] = weighted_f1
    classify_report_dict["auc"] = auc

    # 保留位数
    classify_report_dict = format_dict_numbers(classify_report_dict)

    return classify_report_dict
