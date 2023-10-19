import mido
from mido import MidiFile, MidiTrack, Message


def encode_midi_with_note_events(midi_file_path):
    midi = MidiFile(midi_file_path)
    encoded_events = []

    last_velocity = None
    first_note = False
    interval = 10

    for track in midi.tracks:
        for msg in track:

            time_left = msg.time
            # print("{}: {}".format(msg, time_left))

            while time_left > interval and first_note:

                if time_left >= interval:
                    max_shift = (time_left // interval) * interval
                    encoded_events.append(f'TIME_SHIFT<{max_shift}>')
                    time_left -= max_shift
                else:
                    encoded_events.append(f'TIME_SHIFT<{time_left}>')
                    time_left = 0


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


encoded = encode_midi_with_note_events('muse.mid')
encoding_set = {s for s in encoded}
print(encoding_set)
print(len(encoding_set))
# print(encoded)
decode_events_to_midi(encoded, 'output_midi_file_restored.mid')
