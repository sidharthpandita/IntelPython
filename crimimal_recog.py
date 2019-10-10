import os
import numpy
import sys
import scipy.misc
import genetic
import itertools


# Reading target image to be reproduced using Genetic Algorithm (GA).
target_im = scipy.misc.imread('criminal.jpg')
# Target image after enconding. Value encoding is used.
target_chromosome = GARI.img2chromosome(target_im)

# Population size
sol_per_pop = 8
# Mating pool size
num_parents_mating = 4
# Mutation percentage
mutation_percent = .01

num_possible_permutations = len(list(itertools.permutations(iterable=numpy.arange(0, 
                                                            num_parents_mating), r=2)))
num_required_permutations = sol_per_pop-num_possible_permutations
if(num_required_permutations>num_possible_permutations):
    print(
    )
    sys.exit(1)


# Creating an initial population randomly.
new_population = GARI.initial_population(img_shape=target_im.shape, 
                                         n_individuals=sol_per_pop)

for iteration in range(10000):
    # Measing the fitness of each chromosome in the population.
    qualities = GARI.cal_pop_fitness(target_chromosome, new_population)
    print('Quality : ', numpy.max(qualities), ', Iteration : ', iteration)
    
    # Selecting the best parents in the population for mating.
    parents = GARI.select_mating_pool(new_population, qualities, 
                                      num_parents_mating)
    
    # Generating next generation using crossover.
    new_population = GARI.crossover(parents, target_im.shape, 
                                    n_individuals=sol_per_pop)

    new_population = GARI.mutation(population=new_population,
                                   num_parents_mating=num_parents_mating, 
                                   mut_percent=mutation_percent)
   
    GARI.save_images(iteration, qualities, new_population, target_im.shape, 
                     save_point=500, save_dir=os.curdir+'//')

# Display the final generation
GARI.show_indivs(new_population, target_im.shape)
