from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import embedding
import torch


word2vec_model = embedding.load_model("two_sentence_horror.model")
embedding_matrix = torch.Tensor(word2vec_model.wv.vectors)
vocab_size = len(word2vec_model.wv.index_to_key)

word2idx = {word: idx for idx, word in enumerate(word2vec_model.wv.index_to_key)}
idx2word = {idx: word for word, idx in word2idx.items()}

stories = [
    "She heard a noise. It was getting closer.",
    "I woke up and saw my reflection. It wasn't me.",
    "I turned off the lights. But they turned back on."
]

def story_to_word_indices(story: str) -> torch.Tensor:
    return torch.tensor([word2idx[word] for word in story.split() if word in word2idx])

def collate_fn(batch):
    return pad_sequence(batch, batch_first=True)[:, :-1], pad_sequence(batch, batch_first=True)[:, 1:]

class StoryDataset(Dataset):
    def __init__(self, story_tensors):
        self.story_tensors = story_tensors

    def __len__(self):
        return len(self.story_tensors)

    def __getitem__(self, idx):
        return self.story_tensors[idx]

story_tensors = [story_to_word_indices(story) for story in stories]
train_loader = DataLoader(StoryDataset(story_tensors), batch_size=1, collate_fn=collate_fn, shuffle=True)

