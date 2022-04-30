from mido import MidiFile, MidiTrack, Message, MetaMessage
import mido
import music21
import random
import Notes
import Music
import GA


file = 'input3.mid'

MidiGen = Music.MidiGenerator(file)

tonic, mode, quarters, ticks, tempo, min_note, partition = MidiGen.get_data()

print(tonic, mode)

ChordGen = Notes.ChordGenerator(tonic, mode)

ga = GA.GeneticAlgorithm(ChordGen, MidiGen.chords_count, tonic, mode, ticks, min_note, partition)

epochs = 1
iterations = 10
initial_pop = 10
mutation_prob = 0.8
best = []
sl = []

for i in range(epochs):
    print(i)
    sorted_c, sl = ga.run_ga(initial_pop, mutation_prob, iterations, best)
    best.append(sorted_c[-1])

print(sl)
chords = best[-1].get_genes()
MidiGen.create_new_track(chords)
