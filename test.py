import pandas as pd
import re
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification,BertTokenizer
from torch.utils.data import Dataset

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

class TextDataset(Dataset):
    def __init__(self,texts,tokenizer,max_len):
        self.texts=texts
        self.texts = self.texts.reset_index(drop=True)

        self.tokenizer=tokenizer
        self.max_len=max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):
        text=self.texts[idx]
        encoding=self.tokenizer(text,truncation=True,padding='max_length',max_length=self.max_len,return_tensors='pt')

        return {
            'input_ids':encoding['input_ids'].squeeze(0),
            'attention_mask':encoding['attention_mask'].squeeze(0)
        }


data=pd.read_csv('./data/test.csv',encoding='utf-8')
data['keyword']=data['keyword'].fillna('')
data['location']=data['location'].fillna('')
data['text']=data['keyword']+' '+data['location']+' '+data['text']
data['text']=data['text'].apply(clean_text)

tokenizer=BertTokenizer.from_pretrained('bert-base-cased')

test_dataset=TextDataset(data['text'],tokenizer=tokenizer,max_len=128)
text_loader=torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=True)

device='cuda' if torch.cuda.is_available() else 'cpu'
model=BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=2)
model.load_state_dict(torch.load('./model/bert.pth'))
model.to(device)

model.eval()
all_preds=[]
with torch.no_grad():
    for batch in text_loader:
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)

        outputs=model(input_ids,attention_mask=attention_mask)
        preds=torch.argmax(outputs.logits,dim=-1)
        all_preds.extend(preds.cpu().numpy())
result={
    'id':data['id'],
    'target':all_preds
}
result=pd.DataFrame(result,columns=['id','target'])
result.to_csv('./data/sample_submission.csv',index=False)
print("success test")
