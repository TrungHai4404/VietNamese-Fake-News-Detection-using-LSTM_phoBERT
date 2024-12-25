import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.nn as nn
# Kiểm tra nếu máy tính có GPU thì sẽ dùng GPU (cuda), nếu không thì sử dụng CPU (cpu).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer
# PhoBERT: Mô hình ngôn ngữ cho tiếng Việt. AutoTokenizer dùng để token hóa văn bản, chuyển từ văn bản thành chuỗi số
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Xây dựng một tập dữ liệu
class Custom_Text_Dataset(Dataset):
    def __init__(self, df_dir, tokenizer):  #Đọc file CSV và lưu lại tokenizer
        self.df = pd.read_csv(df_dir)
        self.tokenizer = tokenizer

    def __len__(self):                      #Trả về số lượng dòng trong tập dữ liệu
        return self.df.shape[0]

    def __getitem__(self, idx):             # lấy nhãn và văn bản, Token hóa văn bản, trả về: ids, length, lable
        text = self.df['post_message'][idx]

        label = self.df['label'][idx]
        label = torch.tensor(label, dtype=torch.long)

        tokens = self.tokenizer.tokenize(text)

        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor(ids, dtype=torch.long)

        length = ids.shape
        length = torch.tensor(length, dtype=torch.float32)
        # ids = torch.tensor(ids, dtype=torch.long)
        # Add 0 so that length of all ids is 7180
        # ids = torch.cat((ids, torch.zeros(7180 - len(ids), dtype=torch.long)))

        return ids, length, label


# Tạo các tập dữ liệu train, test, và validation từ các file CSV
train_ds = Custom_Text_Dataset('D:\\fake_news_LSTM_project\\datasets\\train.csv', tokenizer)
test_ds = Custom_Text_Dataset('D:\\fake_news_LSTM_project\\datasets\\test.csv', tokenizer)
valid_ds = Custom_Text_Dataset('D:\\fake_news_LSTM_project\\datasets\\val.csv', tokenizer)



def custom_collate_fn(batch):           #Sắp xếp các mẫu trong batch theo độ dài giảm dần (giúp LSTM xử lý dễ dàng hơn)
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    return sorted_batch

#Tạo các DataLoader để chia dữ liệu thành từng batch (kích thước batch = 64) và sắp xếp chúng.
train_dl = DataLoader(train_ds, batch_size=64, collate_fn=custom_collate_fn)
test_dl = DataLoader(test_ds, batch_size=64, collate_fn=custom_collate_fn)
valid_dl = DataLoader(valid_ds, batch_size = 64, collate_fn = custom_collate_fn)


# Pad các văn bản trong batch để có cùng chiều dài, sau đó lưu batch đã xử lý
training_dl = [ ]
for batch in train_dl:
    X = [torch.tensor(member[0]) for member in batch]
    padded_X = rnn_utils.pad_sequence(X, batch_first=True)
    training_dl.append([padded_X, torch.tensor([member[1] for member in batch]), torch.tensor([member[2] for member in batch])])

testing_dl = [ ]
for batch in test_dl:
    X = [torch.tensor(member[0]) for member in batch]
    padded_X = rnn_utils.pad_sequence(X, batch_first=True)
    testing_dl.append([padded_X, torch.tensor([member[1] for member in batch]), torch.tensor([member[2] for member in batch])])

validing_dl = [ ]
for batch in valid_dl:
    X = [torch.tensor(member[0]) for member in batch]
    padded_X = rnn_utils.pad_sequence(X, batch_first=True)
    validing_dl.append([padded_X, torch.tensor([member[1] for member in batch]), torch.tensor([member[2] for member in batch])])

# Xây dựng lớp LSTMNet
# LSTM dự đoán nhãn tin tức (tin thật/tin giả).
# embedding: Chuyển ID thành vector.
# lstm: LSTM xử lý chuỗi vector.
# fc: Dự đoán nhãn.
# sigmoid: Kích hoạt để tạo xác suất.
class LSTMNet(nn.Module):

    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout):

        super(LSTMNet,self).__init__()

        # Embedding layer converts integer sequences to vector sequences
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        # LSTM layer process the vector sequences
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = True
                           )

        # Dense layer to predict
        self.fc = nn.Linear(hidden_dim * 2,output_dim)
        # Prediction activation function
        self.sigmoid = nn.Sigmoid()


    def forward(self,text,text_lengths):
        embedded = self.embedding(text).to(device)

        # Thanks to packing, LSTM don't see padding tokens
        # and this makes our model better
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)

        packed_output,(hidden_state,cell_state) = self.lstm(packed_embedded)

        # Concatenating the final forward and backward hidden states
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)

        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.sigmoid(dense_outputs)

        return outputs

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
vocab_size = tokenizer.vocab_size 
embedding_dim = 300
hidden_dim = 64
output_dim = 1
n_layers = 2
bidirectional = True
dropout = 0.2

model = LSTMNet(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

import torch.optim as optim
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-4)
criterion = nn.BCELoss()
criterion = criterion.to(device)

# Tính độ chính xác (accuracy) dựa trên nhãn dự đoán.
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc
# Huấn luyện mô hình trên tập train.
def train(model,iterator,optimizer,criterion):

    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()

    for batch in iterator:
        # cleaning the cache of optimizer
        optimizer.zero_grad()

        text,text_lengths = batch[0], batch[1]
        text = text.to(device)
        text_lengths = text_lengths

        # forward propagation and squeezing
        predictions = model(text,text_lengths)

        y_test = batch[2].reshape(-1,1)
        y_test = y_test.to(device)
        # computing loss / backward propagation

        loss = criterion(predictions,y_test.float())
        loss.backward()

        # accuracy
        acc = binary_accuracy(predictions,y_test.float())

        # updating params
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # It'll return the means of loss and accuracy
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Đánh giá mô hình trên tập test/valid
def evaluate(model,iterator,criterion):

    epoch_loss = 0.0
    epoch_acc = 0.0

    # deactivate the dropouts
    model.eval()

    # Sets require_grad flat False
    with torch.no_grad():
        for batch in iterator:
            text,text_lengths = batch[0],batch[1]
            text = text.to(device)

            predictions = model(text,text_lengths)

            #compute loss and accuracy
            y_test = batch[2].reshape(-1,1)
            y_test = y_test.to(device)

            loss = criterion(predictions, y_test.float())

            acc = binary_accuracy(predictions, y_test.float())

            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Huấn luyện mô hình qua 40 epochs và lưu lại mô hình tốt nhất
EPOCH_NUMBER = 40
best_valid = 0.93

for epoch in range(1,EPOCH_NUMBER+1):

    train_loss,train_acc = train(model,training_dl,optimizer,criterion)

    test_loss,test_acc = evaluate(model,testing_dl,criterion)
    valid_loss, valid_acc = evaluate(model, validing_dl, criterion)

    # Showing statistics
    print(f'\tTrain Loss: {train_loss:.3f} | Train. Acc: {train_acc*100:.2f}%')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid. Acc: {valid_acc*100:.2f}%')
    if (valid_acc >= best_valid ):
      best_valid = valid_acc
      torch.save(model.state_dict(), f'best_model_{valid_acc}.pth')
      print(f'Test. Acc: {test_acc*100:.2f}%')
    print()

# Hàm kiểm tra một đoạn văn bản
def input_test(text, tokenizer, model, device):
    tokens = tokenizer.tokenize(text)
    ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long).unsqueeze(0).to(device)
    length = torch.tensor([ids.shape[1]], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(ids, length).item()
    return int(torch.round(torch.tensor(prediction)))

# Dự đoán tin thật/tin giả
input_text = input("Nhập nội dung để kiểm tra: ")
result = input_test(input_text, tokenizer, model, device)
if result == 0:
    print("Đây là tin thật.")
else:
    print("Đây là tin giả.")
