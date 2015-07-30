import numpy as np


def complete_bin(number):
    bit = bin(number)[2:]
    number_of_zero = 12 - len(bit)
    return number_of_zero * '0' + bit


def compute_accuracy(pred_table, joint_probs):
    probs = np.array(map(lambda x: float(x), joint_probs[:,1]))
    pred_probs = np.array(map(lambda x: float(x), pred_table[:,1]))
    return abs(pred_probs - probs).sum()


def query_from_table(query_variable, feature_list, table, *observed):
    negative = 0.0
    positive = 0.0
    index = feature_list.index(query_variable)
    observed_variables = []
    
    for feature, obs in observed:
        feature_index = feature_list.index(feature)
        value = '1' if obs else '0'
        observed_variables.append([feature_index, value])
        
    for bit, prob in table:
        con = False
        for i in observed_variables:
            if bit[i[0]] != i[1]:
                con = True
        if con:
            continue
        if bit[index] == '0':
            negative += float(prob)
        else:
            positive += float(prob)

    return positive / (positive+negative)
