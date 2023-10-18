import torch.nn as nn
import torch
import torch.optim as optim

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_length):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_dim, vocab_size)
        self.max_length = max_length

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        return self.fc(x)


sentences = [
    "Welcome to our end-to-end binary Text-Classification example.",
    "In this demo, we will use the Hugging Faces transformers and datasets library together with a custom Amazon sagemaker-sdk extension to fine-tune a pre-trained transformer on binary text classification",
    "In particular, the pre-trained model will be fine-tuned using the imdb dataset",
    "To get started, we need to set up the environment with a few prerequisite steps, for permissions, configurations, and so on",
]


vocab = set(''.join(sentences))
vocab_size = len(vocab)
token2id = {token: i for i, token in enumerate(vocab)}
id2token = {i: token for token, i in token2id.items()}
print(token2id)
print(id2token)

def generate_text(model, initial_prompt, max_length=100):
    model.eval()
    generated_text = initial_prompt
    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor(generated_text).unsqueeze(0).to(device)
            output = model(input_tensor)
            next_token = output.argmax(dim=-1)[0, -1].item()
            generated_text.append(next_token)
    return generated_text

def tokenize_sentence(sentence):
    return [token2id[char] for char in sentence]


if __name__ == '__main__':
    embed_dim = 32
    num_heads = 2
    num_layers = 2
    max_length = max(map(len, sentences))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(vocab_size, embed_dim, num_heads, num_layers, max_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1000
    for epoch in range(num_epochs):
        total_loss = 0
        for sentence in sentences:
            optimizer.zero_grad()
            tokens = tokenize_sentence(sentence)
            inputs = torch.tensor(tokens[:-1]).unsqueeze(0).to(device)
            targets = torch.tensor(tokens[1:]).unsqueeze(0).to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(0), targets.squeeze(0))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(sentences):.4f}")

    initial_prompt = [0]
    generated_sequence = generate_text(model, initial_prompt)
    generated_sequence = [id2token[id] for id in generated_sequence]
    print(generated_sequence)
