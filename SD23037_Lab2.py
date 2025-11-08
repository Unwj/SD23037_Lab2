import streamlit as st
import numpy as np
from typing import List, Tuple
import random

# Prefer matplotlib for plotting but fall back to plotly if matplotlib isn't available
HAS_MATPLOTLIB = False
HAS_PLOTLY = False
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        HAS_PLOTLY = True
    except Exception:
        HAS_PLOTLY = False

class GeneticAlgorithm:
    def __init__(self, pop_size: int = 300, individual_length: int = 80, 
                 target_ones: int = 50, max_generations: int = 50):
        self.pop_size = pop_size
        self.individual_length = individual_length
        self.target_ones = target_ones
        self.max_generations = max_generations
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def create_individual(self) -> List[int]:
        """Create a random binary string of specified length."""
        return [random.randint(0, 1) for _ in range(self.individual_length)]
    
    def initialize_population(self) -> List[List[int]]:
        """Create initial population of random individuals."""
        return [self.create_individual() for _ in range(self.pop_size)]
    
    def fitness(self, individual: List[int]) -> float:
        """
        Calculate fitness based on how close the number of ones is to target_ones.
        Maximum fitness (80) when number of ones equals target_ones (50).
        """
        num_ones = sum(individual)
        difference = abs(num_ones - self.target_ones)
        return self.individual_length - difference  # Max fitness is 80 when difference is 0
    
    def select_parent(self, population: List[List[int]], fitnesses: List[float]) -> List[int]:
        """Select parent using tournament selection."""
        tournament_size = 3
        tournament = random.sample(list(enumerate(population)), tournament_size)
        return population[max(tournament, key=lambda x: fitnesses[x[0]])[0]]
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform two-point crossover."""
        point1, point2 = sorted(random.sample(range(self.individual_length), 2))
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return child1, child2
    
    def mutate(self, individual: List[int], mutation_rate: float = 0.01) -> List[int]:
        """Perform bit-flip mutation with given probability."""
        return [1-bit if random.random() < mutation_rate else bit for bit in individual]
    
    def evolve(self) -> Tuple[List[int], List[float], List[float]]:
        """Run the genetic algorithm and return the best solution."""
        population = self.initialize_population()
        
        for generation in range(self.max_generations):
            # Calculate fitness for all individuals
            fitnesses = [self.fitness(ind) for ind in population]
            
            # Store statistics
            self.best_fitness_history.append(max(fitnesses))
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
            
            # Create new population
            new_population = []
            elite = population[fitnesses.index(max(fitnesses))]  # Keep best individual
            new_population.append(elite)
            
            while len(new_population) < self.pop_size:
                parent1 = self.select_parent(population, fitnesses)
                parent2 = self.select_parent(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population[:self.pop_size]  # Ensure population size stays constant
        
        # Find best solution
        final_fitnesses = [self.fitness(ind) for ind in population]
        best_individual = population[final_fitnesses.index(max(final_fitnesses))]
        
        return best_individual, self.best_fitness_history, self.avg_fitness_history

# Streamlit UI
st.title('Genetic Algorithm - Bit Pattern Optimization')
st.write("""
### Problem Specifications:
- Population size: 300
- Target number of ones: 50
- Individual length: 80 bits
- Maximum fitness: 80 (when number of ones = 50)
- Number of generations: 50
""")

# Add random seed control
seed = st.sidebar.number_input("Random Seed (for reproducibility)", value=42)
random.seed(seed)

if st.button('Run Genetic Algorithm'):
    ga = GeneticAlgorithm()
    
    # Run evolution with progress bar
    with st.spinner('Running genetic algorithm...'):
        best_solution, best_fitness_history, avg_fitness_history = ga.evolve()
    
    # Display results
    st.subheader('Results')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Ones", sum(best_solution))
    with col2:
        st.metric("Final Fitness", ga.fitness(best_solution))
    with col3:
        st.metric("Solution Length", len(best_solution))
    
    # Plot fitness history
    st.subheader('Fitness History')
    generations = list(range(1, len(best_fitness_history) + 1))
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, best_fitness_history, label='Best Fitness')
        ax.plot(generations, avg_fitness_history, label='Average Fitness')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution Over Generations')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    elif HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=generations, y=best_fitness_history, mode='lines', name='Best Fitness'))
        fig.add_trace(go.Scatter(x=generations, y=avg_fitness_history, mode='lines', name='Average Fitness'))
        fig.update_layout(title='Fitness Evolution Over Generations', xaxis_title='Generation', yaxis_title='Fitness')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No plotting libraries available — showing numeric history')
        st.write({'best': best_fitness_history, 'avg': avg_fitness_history})

    # Display best solution pattern
    st.subheader('Best Solution Bit Pattern')
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(15, 2))
        ax.imshow([best_solution], cmap='binary', aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
    elif HAS_PLOTLY:
        # Plotly heatmap (single-row) — black for 1, white for 0
        z = [best_solution]
        fig = go.Figure(go.Heatmap(z=z, colorscale=[[0, 'white'], [1, 'black']], showscale=False))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(height=150)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.code(''.join(map(str, best_solution)), language='text')
    
    # Additional statistics
    st.subheader('Solution Statistics')
    st.write(f"""
    - Initial best fitness: {best_fitness_history[0]:.2f}
    - Final best fitness: {best_fitness_history[-1]:.2f}
    - Improvement: {(best_fitness_history[-1] - best_fitness_history[0]):.2f}
    - Final number of ones: {sum(best_solution)}
    - Difference from target: {abs(sum(best_solution) - 50)}
    """)
    
    # Display bit pattern as text
    st.subheader('Binary String Representation')
    st.code(''.join(map(str, best_solution)), language='text')