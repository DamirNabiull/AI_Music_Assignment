import random
from mido import MidiFile, MidiTrack, Message
import music21
import numpy as np
import copy
import os.path

# dict() that maps note to midi number
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

# dict() that helps to generate chords for major melody
major = {
    'chords': [0, 2, 4, 5, 7, 9, 11],
    'chord_tonics': ['M', 'm', 'm', 'M', 'M', 'm', 'dim']
}

# dict() that helps to generate chords for minor melody
minor = {
    'chords': [0, 2, 3, 5, 7, 8, 10],
    'chord_tonics': ['m', 'dim', 'M', 'm', 'm', 'M', 'M']
}


# class ChordGenerator that generates notes of major/minor/dim/sus2/sus4 types
class ChordGenerator:
    def __init__(self, note, mode):
        self.note = note
        self.mode = mode
        if mode == 'major':
            self.possible = self.get_all_possible(note, 'M')
        else:
            self.possible = self.get_all_possible(note, 'm')

    # function to get random possible chord for concrete melody (according to mode and tonic)
    def get_possible_chord(self):
        ind = random.randint(0, 6)
        taken_chord = self.possible[ind]
        return taken_chord

    # function to get possible chord for concrete melody by index, where index is a step in table
    def get_possible_chord_ind(self, ind):
        taken_chord = self.possible[ind]
        return taken_chord

    # function to generate sus2 chord
    def get_sus_2(self, main_note):
        return main_note, main_note + 2, main_note + 7

    # function to generate sus4 chord
    def get_sus_4(self, main_note):
        return main_note, main_note + 5, main_note + 7

    # function to generate dim chord
    def get_dim(self, main_note):
        return main_note, main_note + 3, main_note + 6

    # function to generate major chord
    def get_major(self, main_note):
        return main_note, main_note + 4, main_note + 7

    # function to generate minor chord
    def get_minor(self, main_note):
        return main_note, main_note + 3, main_note + 7

    # function to get all possible chords for tonic and mode
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

    # function to generate chord via dicts above
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


# class to work with midi files
class MidiGenerator:
    # __init__ gets all information about midi file
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

    # function that returns information about initial midi file
    def get_data(self):
        return self.tonic, self.mode, self.quarters, self.ticks, self.tempo, self.min_note, self.notes_partition

    # function that creates output midi
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


# class that represents chord
class Chord:
    def __init__(self, start_note, note1, note2, note3, start, end, ind):
        self.type = 'N'
        if ind == 0:
            self.type = 'T'
        elif ind == 3:
            self.type = 'S'
        elif ind == 4:
            self.type = 'D'
        self.start_note = start_note
        self.main_note = note1
        self.note1 = start_note + note1
        self.note2 = start_note + note2
        self.note3 = start_note + note3
        self.start = start
        self.end = end

    # function that changes chord
    def change(self, note1, note2, note3, ind):
        self.type = 'N'
        if ind == 0:
            self.type = 'T'
        elif ind == 3:
            self.type = 'S'
        elif ind == 4:
            self.type = 'D'
        elif ind == 7:
            self.type = 'SUS'
        self.note1 = self.start_note + note1
        self.note2 = self.start_note + note2
        self.note3 = self.start_note + note3

    # function that makes up inversion for chord
    def make_inversion_up(self):
        min_note = self.get_min_note()
        if min_note == self.note1:
            self.note1 += 12
        elif min_note == self.note2:
            self.note2 += 12
        else:
            self.note3 += 12

    # function that makes down inversion for chord
    def make_inversion_down(self):
        max_note = self.get_max_note()
        if max_note == self.note3:
            self.note3 -= 12
        elif max_note == self.note2:
            self.note2 -= 12
        else:
            self.note1 -= 12

    # function that returns max note in chord
    def get_max_note(self):
        return max(self.note1, self.note2, self.note3)

    # function that returns min note in chord
    def get_min_note(self):
        return min(self.note1, self.note2, self.note3)

    # function that returns max note in chord
    def is_note_in_chord(self, note):
        normal_note = note % 12
        if normal_note == self.note1 % 12 or normal_note == self.note2 % 12 or normal_note == self.note3 % 12:
            return True
        return False

    # function that checks if chord has dissonance with note
    def has_dissonance(self, note):
        diss = 0
        t = max(self.note1%12, note%12) - min(self.note1%12, note%12)
        if 0 < t <= 2 or t >= 10:
            diss += 1
        t = max(self.note2%12, note%12) - min(self.note2%12, note%12)
        if 0 < t <= 2 or t >= 10:
            diss += 1
        t = max(self.note3%12, note%12) - min(self.note3%12, note%12)
        if 0 < t <= 2 or t >= 10:
            diss += 1
        return diss


# class Chromosome for Genetic Algorithm
class Chromosome:
    def __init__(self, min_note, ChordGen, gen_len = None, ticks = None, genes = None):
        self.ChordGen = ChordGen
        self.min_note = min_note
        start_from = min_note - min_note % 12 - 12 - np.random.randint(2)*12
        if genes is None:
            self.genes = []
            for i in range(gen_len):
                ind, start_ind = np.random.randint(7), np.random.randint(2)
                note1, note2, note3 = self.ChordGen.get_possible_chord_ind(ind)
                self.genes.append(Chord(start_from, note1%12, note2%12, note3%12, 2*i*ticks, 2*(i+1)*ticks, ind))
        else:
            self.genes = genes

    # function of mutation
    def mutation(self):
        new_chromosome = Chromosome(self.min_note, self.ChordGen, genes=copy.deepcopy(self.genes))
        for chord in new_chromosome.genes:
            p = np.random.uniform(0, 1, 1)
            if p <= 0.1:
                ind = np.random.randint(7)
                note1, note2, note3 = self.ChordGen.get_possible_chord_ind(ind)
                chord.change(note1%12, note2%12, note3%12, ind)
            elif p <= 0.35:
                chord.make_inversion_down()
            elif p <= 0.6:
                chord.make_inversion_up()
            elif p <= 0.8:
                note1, note2, note3 = self.ChordGen.get_sus_2(chord.main_note)
                chord.change(note1, note2, note3, 7)
            else:
                note1, note2, note3 = self.ChordGen.get_sus_4(chord.main_note)
                chord.change(note1, note2, note3, 7)
        return new_chromosome

    # function of crossover
    def crossover(self, other):
        middle = len(self.genes) // 2
        g1 = copy.deepcopy(self.genes)
        g2 = copy.deepcopy(other.genes)
        new_g1 = np.concatenate([g2[:middle], g1[middle:]])
        new_g2 = np.concatenate([g1[:middle], g2[middle:]])
        c1 = Chromosome(self.min_note, self.ChordGen, genes=new_g1)
        c2 = Chromosome(self.min_note, self.ChordGen, genes=new_g2)
        return [c1, c2]

    # function that returns genes
    def get_genes(self):
        return self.genes


# class of Genetic algorithm
class GeneticAlgorithm:
    def __init__(self, ChordGen, chords_count, tonic, mode, ticks, min_note, partition):
        self.ChordGen = ChordGen
        self.chords_count = chords_count
        self.tonic = tonic
        self.mode = mode
        self.ticks = ticks
        self.min_note = min_note
        self.partition = partition
        self.population = []

    # function to add chromosomes to population
    def append_population(self, population):
        self.population = self.population + population

    # function to reset population
    def reset_population(self):
        self.population = []

    # function to generate initial chromosomes in population
    def generate_initial_population(self, c_count):
        for i in range(c_count):
            self.population.append(
                Chromosome(self.min_note, self.ChordGen, gen_len=self.chords_count, ticks=self.ticks)
            )

    # function to mutate chromosomes in population
    def mutation(self, num):
        mutated = []
        mutate_indexes = np.random.randint(0, len(self.population), num)
        for mutate_index in mutate_indexes:
            mutated = mutated + [self.population[mutate_index].mutation()]
        self.population += mutated

    # function to crossover chromosomes in population
    def crossover(self, num):
        crossover_pop = []
        for i in range(num):
            s = list(np.random.randint(0, len(self.population), 2))
            crossover_pop = crossover_pop + self.population[s[0]].crossover(self.population[s[1]])
        self.population += crossover_pop

    # function to get sorted list of chromosomes in population
    def selection(self, num):
        fit_list = [self.fitness(chromosome.get_genes()) for chromosome in self.population]
        sorted_list = sorted(zip(fit_list, self.population), key=lambda f: f[0])
        sorted_chromosomes = [pair[1] for pair in sorted_list]
        sorted_chromosomes = sorted_chromosomes[-num:]
        return sorted_chromosomes

    # function to run algorithm via n iterations
    def run_ga(self, init_population_count, mutation_prob, iterations, pop=None):
        if pop is None:
            pop = []
        self.reset_population()
        self.append_population(pop)
        self.generate_initial_population(init_population_count)
        to_cross = init_population_count + len(pop)
        for i in range(iterations):
            p = np.random.uniform(0, 1, 1)
            self.crossover(to_cross)
            if p < mutation_prob:
                num = to_cross // 2
                self.mutation(num)

        sorted_chromosomes = self.selection(to_cross//2)
        return sorted_chromosomes

    # function that calculate fitness for concrete chords
    def fitness(self, chords):
        c_ind = 0
        total = 0
        for note in self.partition:
            while note['start'] >= chords[c_ind].end:
                if abs(chords[c_ind].get_min_note() - chords[c_ind+1].get_min_note()) > 4:
                    total -= 80 * abs(chords[c_ind].get_min_note() - chords[c_ind+1].get_min_note())

                if chords[c_ind].type == 'S' and chords[c_ind+1].type == 'T':
                    total += 50

                if chords[c_ind].type == 'D' and chords[c_ind+1].type == 'T':
                    total += 50

                if c_ind + 2 > len(chords):
                    if chords[c_ind].type == 'S' and chords[c_ind + 1].type == 'D' and chords[c_ind + 2].type == 'T':
                        total += 100

                s1, s2 = {chords[c_ind].note1, chords[c_ind].note2, chords[c_ind].note3}, {chords[c_ind+1].note1, chords[c_ind+1].note2, chords[c_ind+1].note3}
                s1.intersection(s2)
                if len(s1) > 0:
                    total += 30 * len(s1)
                else:
                    total -= 70

                c_ind += 1

                if note['start'] >= chords[c_ind].end and chords[c_ind].type == 'SUS':
                    total -= 400

            diss = chords[c_ind].has_dissonance(note['note']) > 0

            if note['start'] == chords[c_ind].start:
                if chords[c_ind].is_note_in_chord(note['note']):
                    total += 100
                elif chords[c_ind].type == 'SUS' and diss > 0:
                    total -= 300
                elif diss > 0:
                    total -= 150
            else:
                if chords[c_ind].is_note_in_chord(note['note']):
                    total += 50

                if chords[c_ind].type == 'SUS' and diss > 0:
                    total -= 200
                elif diss > 0:
                    total -= 100

            if chords[c_ind].type == 'SUS' and not chords[c_ind].is_note_in_chord(note['note']):
                total -= 200

            if note['note'] - chords[c_ind].get_max_note() == 0:
                total -= 200

            if note['note'] - chords[c_ind].get_max_note() < 7:
                total -= 200

            if note['note'] - chords[c_ind].get_max_note() > 16:
                total -= 200

        return total


epochs = 25
iterations = 200
initial_pop = 10  # > 1
mutation_prob = 0.8

if __name__ == '__main__':
    print('Print file name (ex: input1.mid) :')
    file = input()
    if os.path.exists(file):
        MidiGen = MidiGenerator(file)
        tonic, mode, quarters, ticks, tempo, min_note, partition = MidiGen.get_data()
        ChordGen = ChordGenerator(tonic, mode)
        ga = GeneticAlgorithm(ChordGen, MidiGen.chords_count, tonic, mode, ticks, min_note, partition)
        best = []
        for i in range(epochs):
            print('Epoch:', i)
            sorted_c = []
            if i == 0:
                sorted_c = ga.run_ga(initial_pop, mutation_prob, iterations, best)
            else:
                sorted_c = ga.run_ga(initial_pop // 2, mutation_prob, iterations, best)
            best = sorted_c[:]
        chords = best[-1].get_genes()
        MidiGen.create_new_track(chords, file[:-4] + '_output')
    else:
        print('File with such name does not exist. Please restart the program and write the correct name.')
