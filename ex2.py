import sys;
import random;
from collections import Counter,defaultdict;
import matplotlib.pyplot as plt;
import numpy as np;

POPULATION_SIZE = 100
NEW_GEN_REPLACE = int(POPULATION_SIZE * 0.15)
MAX_GENERATIONS = 150
MUTATION_RATE = 1
ELITE_SELECTION = 0.2
FITNESS_THRESHOLD = 0.95
LETTERS = "abcdefghijklmnopqrstuvwxyz"
DARWIN_MODE = False 
LAMARCK_MODE = False 

fitness_number = 0

with open("enc.txt", "r") as encFile:
    original_text = encFile.read().strip()
    encoded = original_text.lower().replace(".", "").replace(",", "").replace(";", "")

with open("dict.txt", "r") as dictFile:
    dict = set(dictFile.read().splitlines())

with open("Letter_Freq.txt", "r") as letterFile:
    letterData = {}
    letter_frequencies = letterFile.read().split()
    i = 1
    while i < len(letter_frequencies):
        letterData[letter_frequencies[i].lower()] = float(letter_frequencies[i - 1])
        i += 2

with open("Letter2_Freq.txt", "r") as letterFile:
    letter2_frequencies = {}
    letter_frequencies = letterFile.read().split()
    i = 1
    while i < len(letter_frequencies):
        letter2_frequencies[letter_frequencies[i].lower()] = float(letter_frequencies[i - 1])
        i += 2

def fitness(key):
    global fitness_number
    fitness_number += 1
    decrypted_text = decrypt(key)
    
    letter_freq = Counter(decrypted_text)
    length = len(decrypted_text)
    letter_freq = { char: freq / length for char, freq in letter_freq.items() }

    bigram_freq = defaultdict(int)
    for i in range(length - 1):
        bigram = decrypted_text[i : i + 2]
        bigram_freq[bigram] += 1
    for bigram in bigram_freq:
        bigram_freq[bigram] /= length

    # Create a set to remove duplicates
    decrypted_text_set = set(decrypted_text.split(" "))
    words_in_dict = len(decrypted_text_set.intersection(dict))

    letters_diff = sum([abs(letter_freq[letter] - letter_frequencies[letter]) if letter in letter_freq else 0 for letter in letter_frequencies])
    letters_diff += sum([abs(bigram_freq[bigram] - letter2_frequencies[bigram]) if bigram in bigram_freq else 0 for bigram in letter2_frequencies])

    return (words_in_dict/len(decrypted_text_set)) + (letters_diff/len(decrypted_text))


def decrypt(key, encoded_text = encoded):
    decrypted_text = ""
    for letter in encoded_text:
        if letter in key:
            decrypted_text += key[letter]
        else:
            decrypted_text += letter
    return decrypted_text

def mutate(key):
    index1, index2 = random.choices(LETTERS, k = 2)
    key[index1], key[index2] = key[index2], key[index1]

    return key

def crossover(parent1, parent2):
    # Create a child by randomly selecting half of the letters from each parent
    index = random.randint(0, 25)
    child = {LETTERS[i]: parent1[LETTERS[i]] if i < index else parent2[LETTERS[i]] for i in range(len(LETTERS))}

    # Make sure the child has no duplicate letters
    child_letters = set()
    letters_left = list(LETTERS)
    for key, letter in child.items():
        if letter not in child_letters:
            child_letters.add(letter)
            letters_left.remove(letter)
        else: 
            child[key] = random.choice(letters_left)
            child_letters.add(child[key])
            letters_left.remove(child[key])

    return child


def get_cross_parents(population, fitness, num_of_players = 5):
    parents = []
    for _ in range(2):
        # Select a random sample of players from the population
        players = random.sample(range(len(population)), num_of_players)
        players = sorted(players, key=lambda i: fitness[i], reverse=True)
        parents.append(population[players[0]])
    return parents

def build_generation(population, offspring, fitnesses):
    fitnesses = np.array(fitnesses)
    offspring = np.array(offspring)

    # Find the indices of the keys with the best fitness scores
    sorted_indices = np.argsort(fitnesses)

    # Find the indices of the individuals with the best fitness scores
    # (using -fitness is a trick to get the reverse of the indices)
    best_population = np.argpartition(-fitnesses, NEW_GEN_REPLACE)[:NEW_GEN_REPLACE]

    # Find the indices of the keys with the worst fitness scores
    worst_population = sorted_indices[:3 * NEW_GEN_REPLACE]

    # Calculate the fitness of the offsprings
    fitnesses = np.array([fitness(i) for i in offspring])

    # Find the indices of the best offspring based on fitness scores
    best_offsprings = np.argpartition(-fitnesses, NEW_GEN_REPLACE)[:NEW_GEN_REPLACE]

    # Replace the worst individuals with the best individuals from the population, offspring and the best individuals from the offspring
    for i in range(0, 3 * NEW_GEN_REPLACE, 3):
        population[worst_population[i]] = population[best_population[0]]
        population[worst_population[i + 1]] = population[best_population[int(i / 3)]]
        population[worst_population[i + 2]] = offspring[best_offsprings[int(i / 3)]]

    return population


# The mutation function used in the Darwin and Lamarck modes
def local_optimization(key, best_key):
    random_letter = random.choice(LETTERS)

    # Find the mapping of the random letter in the best solution
    best_key_letter = best_key[random_letter]

    # Find the mapping of the random letter in the current solution
    key_letter = key[random_letter]

    # Swap the letters
    for letter, value in key.items():
        if value == best_key_letter:
            key[random_letter] ,key[letter] = best_key_letter, key_letter
            break

    return key


# The Darwin Evolutionary Algorithm
def darwin(population, fitnesses, best_key):
    
    # Create a mutated version of each key in the population
    mutation_fitnesses = [fitness(local_optimization(key, best_key)) for key in population]

    # Return the best fitness score between the original key and the mutated key
    return [max(fit, mutation_fit) for fit, mutation_fit in zip(fitnesses, mutation_fitnesses)]


# The Lamarck Evolutionary Algorithm
def lamarck(population, fitness_scores, best_key):
    lamarck_fitness = []

    # Create a mutated version of each key in the population
    mutations = [local_optimization(key, best_key) for key in population]
    
    # Find the fitness score of each mutated key
    for i in range(len(population)):
        mutation_fitness = fitness(mutations[i])
        if mutation_fitness > fitness_scores[i]:
            lamarck_fitness.append(mutation_fitness)
            population[i] = mutations[i]
        else:
            lamarck_fitness.append(fitness_scores[i])

    return lamarck_fitness


def genetic_algorithm():
    
    # Generate the initial population
    population = []
    for _ in range(POPULATION_SIZE):
        key = list(LETTERS)
        random.shuffle(list(LETTERS))
        population.append({original: decrypted for original, decrypted in zip(LETTERS, key)})
    
    championships = 0
    best_key = None
    best_key_fitness = 0
    prev_best_fitness = 0

    for generation_idx in range(MAX_GENERATIONS):
        if generation_idx > 0:
            elite_size = int(POPULATION_SIZE * ELITE_SELECTION)
            elite = sorted(range(len(fitnesses)), key=lambda i: -fitnesses[i])[:elite_size]
            next_generation = [population[e] for e in elite]
            for _ in range(len(next_generation), POPULATION_SIZE):
                parent1, parent2 = get_cross_parents(population, fitnesses)
                offspring = crossover(parent1, parent2)
                if random.random() < MUTATION_RATE:
                    offspring = mutate(offspring)
                next_generation.append(offspring)

            # Build the next generation
            population = build_generation(population, next_generation, fitnesses)

        fitnesses = [fitness(key) for key in population]
        if DARWIN_MODE:
            fitnesses = darwin(population, fitnesses, best_key) if best_key else fitnesses
        elif LAMARCK_MODE:
            fitnesses = lamarck(population, fitnesses, best_key) if best_key else fitnesses

        # Find the best key and its fitness score
        best_key_fitness = max(fitnesses)
        best_key = population[fitnesses.index(best_key_fitness)]
        average_fitness = sum(fitnesses) / len(fitnesses)

        
        print(f'Generation {generation_idx}:\tBest Fitness is {best_key_fitness}')
        championships = championships + 1 if best_key_fitness - prev_best_fitness < 0.001 else 0
        prev_best_fitness = best_key_fitness

        # Stop the algorithm if the best fitness score is above the threshold
        if championships == 10:
            break

        plt.scatter(generation_idx, best_key_fitness, color='orange', marker=(5, 1))
        plt.scatter(generation_idx, average_fitness, color='green', marker=(5, 1))

    return best_key, best_key_fitness


if __name__ == "__main__":

    print("Welcome to the Genetic Algorithm Decryption Program")
    if len(sys.argv) > 1:
        if sys.argv[1] in {'-d', '-D', '-darwin', '-Darwin'}:
            DARWIN_MODE = True
            print("* Darwin mode activated *\n")
        elif sys.argv[1] in {'-l', '-L', '-lamarck', '-Lamarck'}:
            LAMARCK_MODE = True
            print("* Lamarck mode activated *\n")
        elif sys.argv[1] in {'-r', '-R', '-regular', '-Regular'}:
            print("* Regular mode activated *\n")
        else:
            print("Not a valid option, exiting...")
            exit()
    else:
        print("* Regular mode activated *\n")

    best_key = None
    best_key_fitness = 0
    number_of_runs = 0


    while best_key_fitness < FITNESS_THRESHOLD and number_of_runs < 5:
        current_key, current_key_fitness = genetic_algorithm()
        number_of_runs += 1
        if current_key_fitness > best_key_fitness:
            best_key_fitness = current_key_fitness
            best_key = current_key
            
    # Decrypt the input text using the best key
    decrypted_text = decrypt(best_key, original_text)

    # Plot the fitness scores
    plt.ylabel('Fitness Scores')
    plt.xlabel('Generation')
    plt.title('Fitness Score per Generation')
    plt.legend(['Best fitness', 'Average fitness'], loc='upper left')
    plt.grid(True)
    plt.savefig('fitness graph.png')

    print(f"\nFinished.\nFitness function has been called {fitness_number} times.\nFiles for Key, plaintext and a graph have been created at the directory.")

    # Write the decrypted text and the permutation to files
    with open("plain.txt", "w") as plain:
        plain.write(decrypted_text)
    with open("perm.txt", "w") as perm:
        for letter, decrypted_letter in best_key.items():
            perm.write(f"{letter}\t{decrypted_letter}\n")