import mido
from mido import MidiFile, MidiTrack, Message

def encode_midi_with_note_events(midi_file_path):
    midi = MidiFile(midi_file_path)
    encoded_events = []

    for track in midi.tracks:
        accumulated_time = 0
        for msg in track:
            accumulated_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                encoded_events.append(('note_on', msg.note, msg.velocity, accumulated_time))
                accumulated_time = 0
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                encoded_events.append(('note_off', msg.note, accumulated_time))
                accumulated_time = 0

    return encoded_events


def decode_events_to_midi(encoded_events, output_midi_path):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    for event in encoded_events:
        if event[0] == 'note_on':
            track.append(Message('note_on', note=event[1], velocity=event[2], time=event[3]))
        elif event[0] == 'note_off':
            track.append(Message('note_off', note=event[1], velocity=0, time=event[2]))

    midi.save(output_midi_path)


encoded = encode_midi_with_note_events('muse.mid')
print(encoded)
decode_events_to_midi(encoded, 'output_midi_file_restored.mid')
