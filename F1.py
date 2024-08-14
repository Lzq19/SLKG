def calculate_f1(precision, recall):  
    if precision == 0 or recall == 0:  
        return 0  
    else:  
        f1 = 2 * (precision * recall) / (precision + recall)  
        return f1  
  
  
if __name__=='__main__':
    recall_value = 0.2366
    precision_value = 0.3104
    
    f1_score = calculate_f1(precision_value, recall_value)  
    
    print(f"F1 Score: {f1_score:.4f}")