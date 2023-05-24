def unlist_dic_value(dic):
    new_val = [v[0] for v in dic.values()]
    new_dic = dict(zip(dic.keys(), new_val))
    
    return new_dic