from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.nn as nn
import embedding
import torch


word2vec_model = embedding.load_model("two_sentence_horror.model")
embedding_matrix = torch.Tensor(word2vec_model.wv.vectors)
vocab_size = len(word2vec_model.wv.index_to_key)

# print(vocab_size)
# print(embedding_matrix)

word2idx = {word: idx for idx, word in enumerate(word2vec_model.wv.index_to_key)}
idx2word = {idx: word for word, idx in word2idx.items()}

stories = [
    "She heard a noise. It was getting closer.",
    "I woke up and saw my reflection. It wasn't me.",
    "I turned off the lights. But they turned back on."
]

def story_to_word_indices(story: str) -> torch.Tensor:
    indices = [word2idx[word] for word in story.split() if word in word2idx]
    
    if len(indices) == 0:
        return None

    return torch.tensor(indices)

def collate_fn(batch):
    batch = [seq for seq in batch if len(seq) > 1]  # Filter out sequences of length <= 1
    
    if len(batch) == 0:
        return None, None
    
    return pad_sequence(batch, batch_first=True)[:, :-1], pad_sequence(batch, batch_first=True)[:, 1:]

class StoryDataset(Dataset):
    def __init__(self, story_tensors):
        self.story_tensors = story_tensors

    def __len__(self):
        return len(self.story_tensors)

    def __getitem__(self, idx):
        return self.story_tensors[idx]


story_tensors = [story_to_word_indices(story) for story in stories]
story_tensors = [tensor for tensor in story_tensors if len(tensor) > 0]
train_loader = DataLoader(StoryDataset(story_tensors), batch_size=1, collate_fn=collate_fn, shuffle=True)


class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, hidden_dim=256, n_layers=2) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.lstm = nn.LSTM(embedding_matrix.size(1), hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        if x == None:
            x = None
            return None, None

        else:
            x = self.embedding(x)

        out, hidden = self.lstm(x, hidden)
        output = self.fc(out)

        print("Hidden in forward : " ,hidden)

        return output, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim))


model = LSTM_model(vocab_size, embedding_matrix)

LOSS_FN = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 50
for epoch in range(epochs):
    model.train()
    
    total_loss = 0
    for inputs, targets in train_loader:
        print("Training inputs : ", inputs)
        print("Training targets : ", targets)
        
        try :
            hidden = model.init_hidden(1)
            output, hidden = model(inputs, hidden)
            outputs = output.view(-1, vocab_size)
            targets = targets.view(-1)

            print("Training outputs : ", outputs)
            print("Training targets : ", targets)

            loss = LOSS_FN(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        except AttributeError:
            pass
    
    print(f'Epoch [{epoch}], Loss: {total_loss/len(train_loader):.4f}')
