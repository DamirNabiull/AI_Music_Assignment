import Notes
import Music
import GA


file = 'input3.mid'

MidiGen = Music.MidiGenerator(file)

tonic, mode, quarters, ticks, tempo, min_note, partition = MidiGen.get_data()

print(tonic, mode)

ChordGen = Notes.ChordGenerator(tonic, mode)

ga = GA.GeneticAlgorithm(ChordGen, MidiGen.chords_count, tonic, mode, ticks, min_note, partition)

epochs = 25
iterations = 200
initial_pop = 10  # >= 1
mutation_prob = 0.8
best = []
sl = []

for i in range(epochs):
    print('Epoch:', i)
    sorted_c = []
    if i == 0:
        sorted_c, sl = ga.run_ga(initial_pop, mutation_prob, iterations, best)
    else:
        sorted_c, sl = ga.run_ga(initial_pop//2, mutation_prob, iterations, best)

    best = sorted_c[-initial_pop//2:]

chords = best[-1].get_genes()
MidiGen.create_new_track(chords, file[:-4] + '_output')
