def calculate_f1(precision, recall):  
    if precision == 0 or recall == 0:  
        return 0  
    else:  
        f1 = 2 * (precision * recall) / (precision + recall)  
        return f1  
  

recall_value = 0.2366  # 召回率  
precision_value = 0.3104  # 精确度  
  
# 计算F1分数  
f1_score = calculate_f1(precision_value, recall_value)  
  
# 输出F1分数  
print(f"F1 Score: {f1_score:.4f}")