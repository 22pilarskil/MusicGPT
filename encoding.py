from mido import MidiFile, MidiTrack, Message, merge_tracks
import os
import json
import copy
import shutil
import pickle
import random



def encode_midi_with_note_events(midi_file_path):
    midi = MidiFile(midi_file_path)
    encoded_events = []

    last_velocity = None
    first_note = False
    granularity = 10

    for msg in merge_tracks(midi.tracks):

        time_left = msg.time

        while time_left > granularity and first_note:

            if time_left >= 1000:
                encoded_events.append('TIME_SHIFT<1000>')
                time_left -= 1000
            else:
                rounded_time = (time_left // granularity) * granularity
                encoded_events.append(f'TIME_SHIFT<{rounded_time}>')
                time_left -= rounded_time


        if msg.type == 'note_on' and msg.velocity > 0:
            first_note = True
            if last_velocity != msg.velocity:
                encoded_events.append(f'SET_VELOCITY<{msg.velocity}>')
                last_velocity = msg.velocity

            encoded_events.append(f'NOTE_ON<{msg.note}>')

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            encoded_events.append(f'NOTE_OFF<{msg.note}>')

    return encoded_events


def decode_events_to_midi(encoded_events, output_midi_path):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    current_velocity = 64
    current_time = 0

    for event in encoded_events:
        if 'SET_VELOCITY' in event:
            current_velocity = int(event.split('<')[1].split('>')[0])

        elif 'TIME_SHIFT' in event:
            shift_amount = int(event.split('<')[1].split('>')[0])
            current_time += shift_amount

        elif 'NOTE_ON' in event:
            note_num = int(event.split('<')[1].split('>')[0])
            track.append(Message('note_on', note=note_num, velocity=current_velocity, time=current_time))
            current_time = 0

        elif 'NOTE_OFF' in event:
            note_num = int(event.split('<')[1].split('>')[0])
            track.append(Message('note_off', note=note_num, velocity=0, time=current_time))
            current_time = 0

    midi.save(output_midi_path)


def split_into_groups(lst, size):
    if len(lst) < size:
        return [lst]
    else:
        chunks = [lst[i:i + size] for i in range(0, len(lst) - size, size)]
        chunks.append(lst[-size:])
        return chunks


def id_to_event(ids):
    dict = json.load(open('config.json'))
    id2token = dict['id2token']
    tokens = []
    for id in ids:
        tokens.append(id2token[str(id)])
    return tokens


def create_dataset_json(input_dir, output_dir, max_dataset_size=None):
    dataset = {
        "tokens": [],
    }
    vocab = set()
    file_list = [f for f in os.listdir(input_dir) if f.endswith(('.midi', '.mid'))]
    total_files = len(file_list)
    count = 0

    # Shuffle the file list to randomize the order of file processing
    random.shuffle(file_list)

    for filename in file_list:
        # Stop if we have reached the max_dataset_size
        if max_dataset_size is not None and len(dataset["tokens"]) >= max_dataset_size:
            break

        filepath = os.path.join(input_dir, filename)
        tokens = encode_midi_with_note_events(filepath)
        token_set = {token for token in tokens}
        vocab = vocab.union(token_set)

        # Check if adding the current file exceeds max_dataset_size
        new_tokens = split_into_groups(tokens, 1000)
        if max_dataset_size is not None and (len(dataset["tokens"]) + len(new_tokens) > max_dataset_size):
            # Add only enough groups to reach the max_dataset_size
            new_tokens = new_tokens[:max_dataset_size - len(dataset["tokens"])]

        dataset["tokens"].extend(new_tokens)

        count += 1
        print(f"Files read: {count} / {total_files} | Points added: {len(dataset['tokens'])}")

    token2id = {}
    for i in range(100):
        token2id[f"TIME_SHIFT<{10*(i+1)}>"] = i
    for i in range(128):
        token2id[f"SET_VELOCITY<{i}>"] = 100 + i
        token2id[f"NOTE_ON<{i}>"] = 100 + i + 128
        token2id[f"NOTE_OFF<{i}>"] = 100 + i + 128 + 128
    id2token = {i: token for token, i in token2id.items()}

    dataset["encodings"] = copy.deepcopy(dataset["tokens"])
    for idx, tokens in enumerate(dataset["tokens"]):
        dataset["encodings"][idx] = [token2id[token] for token in tokens]

    del dataset["tokens"]

    config = {
        "token2id": token2id,
        "id2token": id2token
    }

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    with open(os.path.join(output_dir, "dataset.json"), 'w+') as f:
        json.dump(dataset, f)

    with open(os.path.join(output_dir, "config.json"), 'w+') as f:
        json.dump(config, f)


def create_dataset(output_dir):

    dataset = json.load(open(os.path.join(output_dir, "dataset.json")))
    length = len(dataset['encodings'])
    train_length = int(length * 0.8)
    test_length = int(length * 0.1)
    val_length = int(length * 0.1)
    train = dataset['encodings'][0:train_length]
    test = dataset['encodings'][train_length:train_length + test_length]
    val = dataset['encodings'][train_length + val_length:]

    print(len(train))
    print(len(test))
    print(len(val))

    train_dir = output_dir + '/train'
    test_dir = output_dir + '/test'
    val_dir = output_dir + '/val'
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    os.mkdir(val_dir)

    for i, f in enumerate(train):
        file = open(train_dir + '/train-' + str(i) + '.midi.pickle', 'wb')
        pickle.dump(f, file)
        file.close()

    for i, f in enumerate(test):
        file = open(test_dir + '/test-' + str(i) + '.midi.pickle', 'wb')
        pickle.dump(f, file)
        file.close()

    for i, f in enumerate(val):
        file = open(val_dir + '/val-' + str(i) + '.midi.pickle', 'wb')
        pickle.dump(f, file)
        file.close()


# Function to load data
def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Function to save data
def save_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def create_triplets(jazz_file_path, classical_file_path, output_dir):
    jazz_files = os.listdir(jazz_file_path)
    classical_files = os.listdir(classical_file_path)

    min_length = min(len(jazz_files), len(classical_files))
    jazz_files = jazz_files[:min_length]
    classical_files = classical_files[:min_length]

    random.shuffle(jazz_files)
    random.shuffle(classical_files)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    triplet_count = 0
    for i in range(min_length):
        # Jazz as target and positive, classical as negative
        jazz_target = load_data(os.path.join(jazz_file_path, jazz_files[i]))
        jazz_positive = load_data(os.path.join(jazz_file_path, jazz_files[(i + 1) % min_length]))
        classical_negative = load_data(os.path.join(classical_file_path, classical_files[i]))

        triplet_file_path = os.path.join(output_dir, f'triplet_{triplet_count}.pkl')
        save_data((jazz_target, jazz_positive, classical_negative), triplet_file_path)
        triplet_count += 1

        # Classical as target and positive, jazz as negative
        classical_target = load_data(os.path.join(classical_file_path, classical_files[i]))
        classical_positive = load_data(os.path.join(classical_file_path, classical_files[(i + 1) % min_length]))
        jazz_negative = load_data(os.path.join(jazz_file_path, jazz_files[i]))

        triplet_file_path = os.path.join(output_dir, f'triplet_{triplet_count}.pkl')
        save_data((classical_target, classical_positive, jazz_negative), triplet_file_path)
        triplet_count += 1

        print(f"{triplet_count} /  {min_length * 2}")


if __name__ == '__main__':

    # encoded = encode_midi('/Users/liampilarski/Desktop/MusicGPT/muse.mid')
    # print(len(encoded))
    # encoding_set = {s for s in encoded}
    # # print(encoding_set)
    # # print(len(encoding_set))
    # decode_midi(encoded, 'primer_restored.mid')
    # # decode_events_to_midi([encoded[:250], encoded[250:500]], 'output_midi_file_restored.mid')

    # create_dataset_json("output_jazz", 'dataset_jazz', max_dataset_size=5000)
    # create_dataset("dataset_jazz")
    # create_triplets("dataset_jazz/train", "dataset_maestro/train", "triplets")
    x = load_data("triplets/triplet_1.pkl")
    print(x)
    print(len(x))

    decode_events_to_midi(id_to_event(x[0]), "0.mid")
    decode_events_to_midi(id_to_event(x[1]), "1.mid")
    decode_events_to_midi(id_to_event(x[2]), "2.mid")

