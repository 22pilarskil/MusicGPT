import mido
from mido import MidiFile, MidiTrack, Message
import os
import json
import copy


def encode_midi_with_note_events(midi_file_path):
    midi = MidiFile(midi_file_path)
    encoded_events = []

    last_velocity = None
    first_note = False
    granularity = 10

    for track in midi.tracks:
        for msg in track:

            time_left = msg.time
            # print("{}: {}".format(msg, time_left))

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


directory_path = 'output'
dataset = {
    "tokens": [],
}
vocab = set()
for num, filename in enumerate(os.listdir(directory_path)):
    print(num)
    if filename.endswith(".midi") or filename.endswith(".mid"):

        filepath = os.path.join(directory_path, filename)
        tokens = encode_midi_with_note_events(filepath)
        token_set = {token for token in tokens}
        vocab = vocab.union(token_set)

        dataset["tokens"].extend(split_into_groups(tokens, 1000))


token2id = {token: i for i, token in enumerate(vocab)}
id2token = {i: token for token, i in token2id.items()}

dataset["encodings"] = copy.deepcopy(dataset["tokens"])
for idx, tokens in enumerate(dataset["tokens"]):
    dataset["encodings"][idx] = [token2id[token] for token in tokens]

dataset["token2id"] = token2id
dataset["id2token"] = id2token

with open('dataset.json', 'w+') as f:
    json.dump(dataset, f)
