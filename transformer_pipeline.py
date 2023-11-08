import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_dim)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        attention_out = self.attention(values, keys, queries, mask)
        x = self.norm1(attention_out + queries)
        x = self.dropout(x)
        forward_out = self.feed_forward(x)
        out = self.norm2(forward_out + x)
        return self.dropout(out)

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_length):
        super(MusicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=0.5, forward_expansion=4) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_length = max_length

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.embedding(x)

        mask = self.generate_square_subsequent_mask(seq_length).to(x.device)

        for block in self.blocks:
            out = block(out, out, out, mask)

        return self.fc_out(out)



# Generation Function
def generate_text(model, initial_prompt, max_length=1000):
    model.eval()
    generated_text = initial_prompt
    with torch.no_grad():
        for _ in range(max_length - len(initial_prompt)):
            input_tensor = torch.tensor(generated_text).unsqueeze(0).to(device)
            output = model(input_tensor)
            next_token = output.argmax(dim=-1)[0, -1].item()
            generated_text.append(next_token)
    return generated_text


# Your provided boilerplate code for training starts here

if __name__ == '__main__':

    with open("dataset.json", "r") as file:
        dataset = json.load(file)

    token2id = dataset["token2id"]
    id2token = dataset["id2token"]
    encodings = dataset["encodings"]
    vocab_size = len(dataset["token2id"])

    embed_dim = 32
    num_heads = 2
    num_layers = 2
    max_length = 999

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer(vocab_size, embed_dim, num_heads, num_layers, max_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Define a path to save your model and states
    SAVE_PATH = 'music_transformer_checkpoint.pth'

    start_epoch = 0
    # Check if the checkpoint exists and load it
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded model and optimizer states from epoch {start_epoch}!")

    num_epochs = 500
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for encoding in encodings:
            optimizer.zero_grad()

            half_len = len(encoding) // 2
            inputs = torch.tensor(encoding[:half_len]).unsqueeze(0).to(device)
            targets = torch.tensor(encoding[half_len:]).unsqueeze(0).to(device)

            outputs = model(inputs).transpose(0, 1)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(encodings):.4f}")

        # Save the model and training states after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, SAVE_PATH)
        print(f"Checkpoint saved after epoch {epoch + 1}!")

    initial_prompt = [0]
    generated_sequence = generate_text(model, initial_prompt)
    generated_sequence = [id2token[id] for id in generated_sequence]
    print(generated_sequence)
