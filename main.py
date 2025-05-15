import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,BertForSequenceClassification
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from transformers import get_scheduler
from sklearn.utils import shuffle
from argumation_text import augment_texts
from handle_data import clean_text


class TextDataset(Dataset):
    def __init__(self,texts,targets,tokenizer,max_len):
        self.texts=texts
        self.targets=targets
        self.texts = self.texts.reset_index(drop=True)
        self.targets = self.targets.reset_index(drop=True)

        self.tokenizer=tokenizer
        self.max_len=max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):
        text=self.texts[idx]
        target=self.targets[idx]
        encoding=self.tokenizer(text,truncation=True,padding='max_length',max_length=self.max_len,return_tensors='pt')

        return {
            'input_ids':encoding['input_ids'].squeeze(0),
            'attention_mask':encoding['attention_mask'].squeeze(0),
            'labels':torch.tensor(target,dtype=torch.long)
        }


data=pd.read_csv('./data/train.csv',encoding='utf-8')
data['text']=data['text'].apply(clean_text)
x_train,x_test,y_train,y_test=train_test_split(data['text'],data['target'],test_size=0.2)

augmented_df = augment_texts(x_train, y_train, times=2)

# 合并原始与增强数据
x_train_final = pd.concat([x_train, augmented_df['text']], ignore_index=True)
y_train_final = pd.concat([y_train, augmented_df['target']], ignore_index=True)
x_train_final,y_train_final=shuffle(x_train_final,y_train_final)

tokenizer=BertTokenizer.from_pretrained('bert-base-cased')

train_dataset=TextDataset(x_train_final,y_train_final,tokenizer,max_len=160)
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)
test_dataset=TextDataset(x_test,y_test,tokenizer,max_len=160)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False)

device='cuda' if torch.cuda.is_available() else 'cpu'

model=BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=2)
optimizer=Adam(model.parameters(),lr=5e-5)
model.to(device)

epoches=10
scheduler=get_scheduler('linear',optimizer=optimizer,num_warmup_steps=0, num_training_steps=len(train_loader)*epoches)
for epoch in range(epoches):
    model.train()
    total_loss=0.0
    for batch in train_loader:
        input_id=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)

        optimizer.zero_grad()
        outputs=model(input_ids=input_id,attention_mask=attention_mask,labels=labels)
        loss=outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss+=loss.item()

    avg_loss=total_loss/len(train_loader)
    print(f"Epoch {epoch+1}/{epoches}: Training_loss={avg_loss:.4f}")

model.eval()
all_preds=[]
all_labels=[]
with torch.no_grad():
    for batch in test_loader:
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)

        outputs=model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        logits=outputs.logits
        preds=torch.argmax(logits,dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy=accuracy_score(all_labels,all_preds)
print("Accuracy:",accuracy)


torch.save(model.state_dict(),'./model/bert.pth')
print("Model saved")