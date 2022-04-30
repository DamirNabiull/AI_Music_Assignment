import random

notes = {
    'C': 0,
    'C#': 1,
    'D-': 1,
    'D': 2,
    'D#': 3,
    'E-': 3,
    'E': 4,
    'F-': 4,
    'F': 5,
    'E#': 5,
    'F#': 6,
    'G-': 6,
    'G': 7,
    'G#': 8,
    'A-': 8,
    'A': 9,
    'A#': 10,
    'B-': 10,
    'B': 11,
}

major = {
    'chords': [0, 2, 4, 5, 7, 9, 11],
    'chord_tonics': ['M', 'm', 'm', 'M', 'M', 'm', 'dim']
}

minor = {
    'chords': [0, 2, 3, 5, 7, 8, 10],
    'chord_tonics': ['m', 'dim', 'M', 'm', 'm', 'M', 'M']
}


class ChordGenerator:
    def __init__(self, note, mode):
        self.note = note
        self.mode = mode
        if mode == 'major':
            self.possible = self.get_all_possible(note, 'M')
        else:
            self.possible = self.get_all_possible(note, 'm')

    def get_possible_chord(self):
        ind = random.randint(0, 6)
        taken_chord = self.possible[ind]
        return taken_chord

    def get_possible_chord_ind(self, ind):
        taken_chord = self.possible[ind]
        return taken_chord

    def get_sus_2(self, main_note):
        return main_note, main_note + 2, main_note + 7

    def get_sus_4(self, main_note):
        return main_note, main_note + 5, main_note + 7

    def get_dim(self, main_note):
        return main_note, main_note + 3, main_note + 6

    def get_major(self, main_note):
        return main_note, main_note + 4, main_note + 7

    def get_minor(self, main_note):
        return main_note, main_note + 3, main_note + 7

    def get_all_possible(self, tonic, mode):
        tonic_note = notes[tonic]
        chords = []
        chord_tonics = []
        if mode == 'M':
            chords = major['chords']
            chord_tonics = major['chord_tonics']
        else:
            chords = minor['chords']
            chord_tonics = minor['chord_tonics']
        return self.get_chords(tonic_note, chords, chord_tonics)

    def get_chords(self, tonic_note, chords, chord_tonics):
        to_return = []
        for i in range(len(chords)):
            main_chord_note = (tonic_note + chords[i]) % 12
            if chord_tonics[i] == 'M':
                to_return.append(self.get_major(main_chord_note))
            elif chord_tonics[i] == 'm':
                to_return.append(self.get_minor(main_chord_note))
            else:
                to_return.append(self.get_dim(main_chord_note))

        to_return = to_return
        return to_return
