import torch.nn as nn
import torch
import gensim
import embedding

class model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        print("looped")

        self.RNN = nn.RNN(10, 20, 2, batch_first=True)
        self.FC = nn.Linear(20, 5)

    def forward(self, x) -> None:
        output, hidden_layer = self.RNN(x)
        # print("HIDDEN LAYERS : ", hidden_layer)
        # print("UNSHAPED OUTPUT : ", output)
        ShapedOutput = self.FC(output)

        return ShapedOutput

# RNN_model = model()
# x = torch.randn(3, 10, 10)
#
# output = RNN_model(x)

gensim_model = embedding.load_model("two_sentence_horror.model")

vocab_size = len(gensim_model.wv.index_to_key)  # Number of words in the vocabulary
embedding_dim = gensim_model.vector_size        # Embedding size (usually 100, 200, etc.)
print(vocab_size)
# Create a matrix of size (vocab_size, embedding_dim) to store the word vectors
embedding_matrix = torch.zeros((vocab_size, embedding_dim))

# Populate the embedding matrix with word vectors from Gensim
for i, word in enumerate(gensim_model.wv.index_to_key):
    embedding_matrix[i] = torch.tensor(gensim_model.wv[word])

print(embedding_matrix)
