import numpy as np
import pandas as pd
import re

def remove_html_tags(text):
    # 去除HTML标签
    return re.sub(r'<.*?>', '', text)

def remove_urls(text):
    # 去除URL
    return re.sub(r'http[s]?://\S+|www\.\S+', '', text)

def remove_special_chars(text):
    # 去除特殊字符
    return re.sub(r'[^A-Za-z0-9\s]+', '', text)

def clean_text(text):
    # 综合清洗步骤
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    return text


# data=pd.read_csv('./data/train.csv',encoding='utf-8')
# data['keyword']=data['keyword'].fillna('')
# data['location']=data['location'].fillna('')
# data['text']=data['keyword'].astype(str)+' '+data['location'].astype(str)+' '+data['text'].astype(str)
# data['text']=data['text'].apply(clean_text)
# data.to_csv('./data/train2.csv',index=False)





