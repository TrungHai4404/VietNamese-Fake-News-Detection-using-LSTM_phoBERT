from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer
import torch.nn as nn
import os
# Tạo ứng dụng Flask
app = Flask(__name__)

# Cấu hình thiết bị (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load mô hình và tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

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

vocab_size = tokenizer.vocab_size 
embedding_dim = 300
hidden_dim = 64
output_dim = 1
n_layers = 2
bidirectional = True
dropout = 0.2
model = LSTMNet(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
# Đường dẫn tới mô hình
model_path ='D:/fake_news_LSTM_project/model/model_0.93_phobert.pth'

# Tải mô hình, ánh xạ sang CPU nếu cần
model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))

model = model.to(device)

# Hàm kiểm tra một đoạn văn bản
def input_text(text, tokenizer, model, device):
    tokens = tokenizer.tokenize(text)
    ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long).unsqueeze(0).to(device)
    length = torch.tensor([ids.shape[1]], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(ids, length).item()
    return int(torch.round(torch.tensor(prediction)))


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']  # Nhận văn bản từ form
        result = input_text(text, tokenizer, model, device)
        if int(result) == 0:
            prediction = "Đây là tin thật"
        else:
            prediction = "Đây là tin giả"
        return render_template('index.html', result=prediction)
    return render_template('index.html', result="")

if __name__ == '__main__':
    app.run(debug=True)
