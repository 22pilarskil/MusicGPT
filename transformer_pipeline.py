import torch.optim as optim
import torch.nn as nn
import torch
import json
from models import MusicTransformer

with open("dataset.json", "r") as file:
    dataset = json.load(file)

token2id = dataset["token2id"]
id2token = dataset["id2token"]
encodings = dataset["encodings"]
vocab_size = len(dataset["token2id"])
print(vocab_size)


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


if __name__ == '__main__':
    embed_dim = 32
    num_heads = 2
    num_layers = 2
    max_length = max(map(len, encodings))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer(vocab_size, embed_dim, num_heads, num_layers, max_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 500
    for epoch in range(num_epochs):
        total_loss = 0
        for encoding in encodings:
            optimizer.zero_grad()
            inputs = torch.tensor(encoding[:-1]).unsqueeze(0).to(device)
            targets = torch.tensor(encoding[1:]).unsqueeze(0).to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(0), targets.squeeze(0))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(encodings):.4f}")

    torch.save(model, 'music_transformer.pth')

    initial_prompt = [0]
    generated_sequence = generate_text(model, initial_prompt)
    generated_sequence = [id2token[id] for id in generated_sequence]
    print(generated_sequence)
