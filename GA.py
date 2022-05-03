import numpy as np
import copy


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

    def make_inversion_up(self):
        min_note = self.get_min_note()
        if min_note == self.note1:
            self.note1 += 12
        elif min_note == self.note2:
            self.note2 += 12
        else:
            self.note3 += 12

    def make_inversion_down(self):
        max_note = self.get_max_note()
        if max_note == self.note3:
            self.note3 -= 12
        elif max_note == self.note2:
            self.note2 -= 12
        else:
            self.note1 -= 12

    def get_max_note(self):
        return max(self.note1, self.note2, self.note3)

    def get_min_note(self):
        return min(self.note1, self.note2, self.note3)

    def is_note_in_chord(self, note):
        normal_note = note % 12
        if normal_note == self.note1 % 12 or normal_note == self.note2 % 12 or normal_note == self.note3 % 12:
            return True
        return False

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

    def crossover(self, other):
        middle = len(self.genes) // 2
        g1 = copy.deepcopy(self.genes)
        g2 = copy.deepcopy(other.genes)
        new_g1 = np.concatenate([g2[:middle], g1[middle:]])
        new_g2 = np.concatenate([g1[:middle], g2[middle:]])
        c1 = Chromosome(self.min_note, self.ChordGen, genes=new_g1)
        c2 = Chromosome(self.min_note, self.ChordGen, genes=new_g2)
        return [c1, c2]

    def get_genes(self):
        return self.genes


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

    def append_population(self, population):
        self.population = self.population + population

    def reset_population(self):
        self.population = []

    def generate_initial_population(self, c_count):
        for i in range(c_count):
            self.population.append(
                Chromosome(self.min_note, self.ChordGen, gen_len=self.chords_count, ticks=self.ticks)
            )

    def mutation(self, num):
        mutated = []
        mutate_indexes = np.random.randint(0, len(self.population), num)
        for mutate_index in mutate_indexes:
            mutated = mutated + [self.population[mutate_index].mutation()]
        self.population += mutated

    def crossover(self, num):
        crossover_pop = []
        for i in range(num):
            s = list(np.random.randint(0, len(self.population), 2))
            crossover_pop = crossover_pop + self.population[s[0]].crossover(self.population[s[1]])
        self.population += crossover_pop

    def calc_fitness(self):
        fitness_list = [self.fitness(chromosome.get_genes()) for chromosome in self.population]
        sorted_list = sorted(zip(fitness_list, self.population), key=lambda f: f[0])
        sorted_chromosomes = [pair[1] for pair in sorted_list]
        self.population = sorted_chromosomes
        return sorted_chromosomes, sorted_list

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

        sorted_chromosomes, sl = self.calc_fitness()
        return sorted_chromosomes, sl

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


