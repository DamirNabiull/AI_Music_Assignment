from mido import MidiFile, MidiTrack, Message, MetaMessage
import mido
import music21


class MidiGenerator:
    def __init__(self, file_name):
        self.file_name = file_name
        file = music21.converter.parse(file_name)
        key = file.analyze('key')
        tonic = key.tonic.name
        mode = key.mode
        mid = MidiFile(file_name, clip=True)
        tempo = 0
        velocity = 0
        total_ticks = 0
        min_note = 1000
        notes = []

        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                elif msg.type == 'note_on' or msg.type == 'note_off':
                    min_note = min(min_note, msg.note)
                    total_ticks += msg.time
                    if msg.type == 'note_on':
                        velocity += msg.velocity
                        notes.append({'start': total_ticks, 'end': 100000, 'note': msg.note})
                    else:
                        for i in range(len(notes) - 1, -1, -1):
                            if notes[i]['note'] == msg.note:
                                notes[i]['end'] = total_ticks
                                break

        self.avg_velocity = (velocity // len(notes)) - 10

        ticks = mid.ticks_per_beat
        quarters = total_ticks // ticks

        self.chords_count = quarters // 2
        if quarters % 2 > 0:
            self.chords_count += 1

        self.tonic = tonic
        self.mode = mode
        self.quarters = quarters
        self.ticks = ticks
        self.tempo = tempo
        self.min_note = min_note
        self.notes_partition = notes

    def get_data(self):
        return self.tonic, self.mode, self.quarters, self.ticks, self.tempo, self.min_note, self.notes_partition

    def create_new_track(self, chords, track_name='new_song'):
        mid = MidiFile(self.file_name, clip=True)
        new_mid = MidiFile(type=1)
        new_mid.ticks_per_beat = self.ticks

        for track in mid.tracks:
            new_mid.tracks.append(track)

        track1 = MidiTrack()
        track2 = MidiTrack()
        track3 = MidiTrack()
        new_mid.tracks.append(track1)
        new_mid.tracks.append(track2)
        new_mid.tracks.append(track3)
        half = 2 * self.ticks

        for gen in chords:
            track1.append(Message('note_on', note=gen.note1, velocity=self.avg_velocity, time=0))
            track1.append(Message('note_off', note=gen.note1, velocity=self.avg_velocity, time=half))
            track2.append(Message('note_on', note=gen.note2, velocity=self.avg_velocity, time=0))
            track2.append(Message('note_off', note=gen.note2, velocity=self.avg_velocity, time=half))
            track3.append(Message('note_on', note=gen.note3, velocity=self.avg_velocity, time=0))
            track3.append(Message('note_off', note=gen.note3, velocity=self.avg_velocity, time=half))

        new_mid.save(f'{track_name}.mid')


