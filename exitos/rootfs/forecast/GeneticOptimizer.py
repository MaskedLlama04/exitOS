import logging
import numpy as np
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass

logger = logging.getLogger("exitOS")


@dataclass
class GeneticAlgorithmConfig:
    """Configuració per a l'algorisme genètic"""
    population_size: int = 150
    max_generations: int = 150
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    elite_size: int = 10
    tournament_size: int = 5
    convergence_threshold: float = 0.0001


class GeneticOptimizer:
    """
    Optimitzador basat en Algorisme Genètic per a la planificació energètica.
    Implementa una funció objectiu universal que funciona amb qualsevol configuració de dispositius.
    """

    def __init__(self, config: GeneticAlgorithmConfig = None):
        """
        Inicialitza l'optimitzador genètic.
        
        Args:
            config: Configuració de l'algorisme genètic. Si és None, s'utilitza la configuració per defecte.
        """
        self.config = config if config is not None else GeneticAlgorithmConfig()
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation = 0

    def optimize(
        self,
        bounds: List[Tuple[float, float]],
        objective_function: Callable,
        integrality: List[int] = None,
        callback: Callable = None
    ) -> Dict:
        """
        Executa l'optimització utilitzant l'algorisme genètic.
        
        Args:
            bounds: Llista de tuples (min, max) per a cada variable de decisió
            objective_function: Funció a minimitzar que rep un array de configuració
            integrality: Llista de 0/1 indicant si cada variable és entera (1) o contínua (0)
            callback: Funció opcional que es crida després de cada generació
            
        Returns:
            Diccionari amb els resultats de l'optimització:
                - 'x': Millor solució trobada
                - 'fun': Valor de la funció objectiu per a la millor solució
                - 'nfev': Nombre d'avaluacions de la funció
                - 'nit': Nombre d'iteracions (generacions)
                - 'message': Missatge de finalització
        """
        logger.info(f"🧬 Iniciant optimització amb Algorisme Genètic")
        logger.info(f"   ▫️ Mida població: {self.config.population_size}")
        logger.info(f"   ▫️ Generacions màximes: {self.config.max_generations}")
        logger.info(f"   ▫️ Taxa mutació: {self.config.mutation_rate}")
        logger.info(f"   ▫️ Taxa encreuament: {self.config.crossover_rate}")

        # Inicialització
        self.bounds = bounds
        self.integrality = integrality if integrality is not None else [0] * len(bounds)
        self.n_vars = len(bounds)
        self.n_evaluations = 0

        # Crear població inicial
        population = self._initialize_population()

        # Avaluar població inicial
        fitness = np.array([objective_function(ind) for ind in population])
        self.n_evaluations += len(population)

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        self.best_fitness_history = [best_fitness]
        self.avg_fitness_history = [np.mean(fitness)]

        logger.info(f"   ▫️ Fitness inicial millor: {best_fitness:.4f}")
        logger.info(f"   ▫️ Fitness inicial mitjà: {np.mean(fitness):.4f}")

        # Bucle principal de generacions
        for generation in range(self.config.max_generations):
            self.generation = generation

            # Selecció
            parents = self._selection(population, fitness)

            # Crear nova generació
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]

                # Encreuament
                if np.random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutació
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)

                offspring.extend([child1, child2])

            # Avaluar descendents
            offspring_fitness = np.array([objective_function(ind) for ind in offspring])
            self.n_evaluations += len(offspring)

            # Elitisme: mantenir els millors individus
            elite_indices = np.argsort(fitness)[:self.config.elite_size]
            elite_population = [population[i] for i in elite_indices]
            elite_fitness = fitness[elite_indices]

            # Combinar elits amb descendents
            population = elite_population + offspring[:self.config.population_size - self.config.elite_size]
            fitness = np.concatenate([elite_fitness, offspring_fitness[:self.config.population_size - self.config.elite_size]])

            # Actualitzar millor solució
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[current_best_idx].copy()
                logger.debug(f"   ✨ Nova millor solució a generació {generation}: {best_fitness:.4f}")

            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness))

            # Callback
            if callback is not None:
                callback(bounds, self._calculate_convergence())

            # Criteri de convergència
            if generation > 10:
                recent_improvement = abs(self.best_fitness_history[-10] - best_fitness)
                if recent_improvement < self.config.convergence_threshold:
                    logger.info(f"   ✅ Convergència assolida a generació {generation}")
                    break

            # Log cada 10 generacions
            if generation % 10 == 0:
                logger.info(f"   🔄 Generació {generation}: Millor={best_fitness:.4f}, Mitjà={np.mean(fitness):.4f}")

        logger.info(f"🏁 Optimització finalitzada")
        logger.info(f"   ▫️ Millor fitness: {best_fitness:.4f}")
        logger.info(f"   ▫️ Generacions: {generation + 1}")
        logger.info(f"   ▫️ Avaluacions totals: {self.n_evaluations}")

        return {
            'x': best_solution,
            'fun': best_fitness,
            'nfev': self.n_evaluations,
            'nit': generation + 1,
            'message': 'Optimization terminated successfully.'
        }

    def _initialize_population(self) -> List[np.ndarray]:
        """
        Crea la població inicial de manera aleatòria dins dels límits especificats.
        
        Returns:
            Llista d'individus (arrays numpy)
        """
        population = []
        for _ in range(self.config.population_size):
            individual = np.zeros(self.n_vars)
            for i, (lower, upper) in enumerate(self.bounds):
                if self.integrality[i] == 1:
                    # Variable entera
                    individual[i] = np.random.randint(int(lower), int(upper) + 1)
                else:
                    # Variable contínua
                    individual[i] = np.random.uniform(lower, upper)
            population.append(individual)
        return population

    def _selection(self, population: List[np.ndarray], fitness: np.ndarray) -> List[np.ndarray]:
        """
        Selecciona individus per a la reproducció utilitzant selecció per torneig.
        
        Args:
            population: Població actual
            fitness: Valors de fitness per a cada individu
            
        Returns:
            Llista d'individus seleccionats
        """
        selected = []
        for _ in range(self.config.population_size):
            # Selecció per torneig
            tournament_indices = np.random.choice(
                len(population),
                size=self.config.tournament_size,
                replace=False
            )
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return selected

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realitza l'encreuament entre dos pares utilitzant encreuament uniforme.
        
        Args:
            parent1: Primer pare
            parent2: Segon pare
            
        Returns:
            Tupla amb dos fills
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Encreuament uniforme
        mask = np.random.random(self.n_vars) < 0.5
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]

        return child1, child2

    def _mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        Aplica mutació a un individu.
        
        Args:
            individual: Individu a mutar
            
        Returns:
            Individu mutat
        """
        mutated = individual.copy()

        for i in range(self.n_vars):
            if np.random.random() < self.config.mutation_rate:
                lower, upper = self.bounds[i]

                if self.integrality[i] == 1:
                    # Mutació per a variables enteres: canvi aleatori dins del rang
                    mutated[i] = np.random.randint(int(lower), int(upper) + 1)
                else:
                    # Mutació gaussiana per a variables contínues
                    sigma = (upper - lower) * 0.1  # 10% del rang
                    mutated[i] = individual[i] + np.random.normal(0, sigma)
                    # Assegurar que està dins dels límits
                    mutated[i] = np.clip(mutated[i], lower, upper)

        return mutated

    def _calculate_convergence(self) -> float:
        """
        Calcula una mètrica de convergència basada en la millora recent.
        
        Returns:
            Valor de convergència
        """
        if len(self.best_fitness_history) < 2:
            return 1.0
        return abs(self.best_fitness_history[-1] - self.best_fitness_history[-2])


class UniversalObjectiveFunction:
    """
    Funció objectiu universal que funciona amb qualsevol configuració de dispositius.
    Calcula el cost total considerant tots els consumidors, generadors i emmagatzematge d'energia.
    """

    def __init__(
        self,
        consumers: Dict,
        generators: Dict,
        energy_storages: Dict,
        electricity_prices: List[float],
        global_consumer_forecast: Dict,
        global_generator_forecast: Dict,
        horizon: int,
        horizon_min: int
    ):
        """
        Inicialitza la funció objectiu universal.
        
        Args:
            consumers: Diccionari de dispositius consumidors
            generators: Diccionari de dispositius generadors
            energy_storages: Diccionari de dispositius d'emmagatzematge
            electricity_prices: Preus de l'electricitat per hora
            global_consumer_forecast: Previsió de consum global
            global_generator_forecast: Previsió de generació global
            horizon: Horitzó de planificació (hores)
            horizon_min: Intervals per hora
        """
        self.consumers = consumers
        self.generators = generators
        self.energy_storages = energy_storages
        self.electricity_prices = electricity_prices
        self.global_consumer_forecast = global_consumer_forecast
        self.global_generator_forecast = global_generator_forecast
        self.horizon = horizon
        self.horizon_min = horizon_min

        self.current_cost = 0
        self.current_balance = None
        self.current_penalty_cost = 0  # Track penalty costs from devices

    def __call__(self, config: np.ndarray) -> float:
        """
        Calcula el cost total per a una configuració donada.
        
        Args:
            config: Vector de configuració per a tots els dispositius
            
        Returns:
            Cost total (a minimitzar) = cost electricitat + penalitzacions
        """
        # Reset penalty cost for this evaluation
        self.current_penalty_cost = 0
        
        total_balance = self._calculate_total_balance(config)
        electricity_cost = self._calculate_electricity_cost(total_balance)
        
        # Total cost includes both electricity and penalty costs
        total_cost = electricity_cost + self.current_penalty_cost

        self.current_cost = total_cost
        self.current_balance = total_balance

        return total_cost

    def _calculate_total_balance(self, config: np.ndarray) -> List[float]:
        """
        Calcula el balanç energètic total per a cada interval de temps.
        
        Args:
            config: Vector de configuració
            
        Returns:
            Llista amb el balanç energètic per cada interval
        """
        num_intervals = self.horizon * self.horizon_min
        total_balance = [0.0] * num_intervals

        # Calcular consum total dels consumidors controlables
        total_consumers = self._calculate_consumer_balance(config)

        # Calcular generació total dels generadors controlables
        total_generators = self._calculate_generator_balance(config)

        # Afegir previsions globals (consum i generació no controlables)
        for hour in range(num_intervals):
            total_consumers[hour] += self.global_consumer_forecast['forecast_data'][hour]
            total_generators[hour] += self.global_generator_forecast['forecast_data'][hour]

            # Balanç = Consum - Generació (positiu = comprar, negatiu = vendre)
            total_balance[hour] = total_consumers[hour] - total_generators[hour]

        # Aplicar emmagatzematge d'energia
        total_balance = self._calculate_energy_storage_balance(config, total_balance)

        return total_balance

    def _calculate_consumer_balance(self, config: np.ndarray) -> List[float]:
        """
        Calcula el consum total de tots els consumidors controlables.
        També acumula les penalitzacions dels dispositius.
        
        Args:
            config: Vector de configuració
            
        Returns:
            Llista amb el consum per cada interval
        """
        num_intervals = self.horizon * self.horizon_min
        total_consumption = [0.0] * num_intervals

        for consumer in self.consumers.values():
            start = consumer.vbound_start
            end = consumer.vbound_end

            # Simular el dispositiu amb la seva part de la configuració
            result = consumer.simula(config[start:end + 1].copy(), self.horizon, self.horizon_min)

            # Sumar el perfil de consum
            for hour in range(min(len(result['consumption_profile']), num_intervals)):
                total_consumption[hour] += result['consumption_profile'][hour]
            
            # Acumular penalitzacions del dispositiu
            if 'total_cost' in result:
                self.current_penalty_cost += result['total_cost']

        return total_consumption

    def _calculate_generator_balance(self, config: np.ndarray) -> List[float]:
        """
        Calcula la generació total de tots els generadors controlables.
        També acumula les penalitzacions dels dispositius.
        
        Args:
            config: Vector de configuració
            
        Returns:
            Llista amb la generació per cada interval
        """
        num_intervals = self.horizon * self.horizon_min
        total_generation = [0.0] * num_intervals

        for generator in self.generators.values():
            start = generator.vbound_start
            end = generator.vbound_end

            # Simular el dispositiu amb la seva part de la configuració
            result = generator.simula(config[start:end + 1].copy(), self.horizon, self.horizon_min)

            # Sumar el perfil de generació
            for hour in range(min(len(result['consumption_profile']), num_intervals)):
                total_generation[hour] += result['consumption_profile'][hour]
            
            # Acumular penalitzacions del dispositiu
            if 'total_cost' in result:
                self.current_penalty_cost += result['total_cost']

        return total_generation

    def _calculate_energy_storage_balance(
        self,
        config: np.ndarray,
        total_balance: List[float]
    ) -> List[float]:
        """
        Aplica l'efecte dels sistemes d'emmagatzematge d'energia al balanç total.
        També acumula les penalitzacions dels dispositius.
        
        Args:
            config: Vector de configuració
            total_balance: Balanç energètic abans de l'emmagatzematge
            
        Returns:
            Balanç energètic actualitzat amb l'emmagatzematge
        """
        updated_balance = list(total_balance)
        num_intervals = len(total_balance)

        for energy_storage in self.energy_storages.values():
            start = energy_storage.vbound_start
            end = energy_storage.vbound_end

            # Simular el dispositiu d'emmagatzematge
            result = energy_storage.simula(config[start:end + 1], self.horizon, self.horizon_min)

            # Afegir el perfil de consum/descàrrega de la bateria
            for hour in range(min(len(result['consumption_profile']), num_intervals)):
                updated_balance[hour] += result['consumption_profile'][hour]
            
            # Acumular penalitzacions del dispositiu
            if 'total_cost' in result:
                self.current_penalty_cost += result['total_cost']

        return updated_balance

    def _calculate_electricity_cost(self, total_balance: List[float]) -> float:
        """
        Calcula el cost econòmic de l'electricitat basant-se en el balanç energètic i els preus.
        Nota: Aquest mètode només calcula el cost de l'electricitat, no les penalitzacions.
        
        Args:
            total_balance: Balanç energètic per cada interval (en W)
            
        Returns:
            Cost de l'electricitat en unitats monetàries
        """
        electricity_cost = 0.0

        for hour in range(len(total_balance)):
            # Cost = Balanç (W) * Preu (€/kWh) / 1000 (conversió W -> kW)
            # Balanç positiu = comprar electricitat (cost)
            # Balanç negatiu = vendre electricitat (ingrés)
            electricity_cost += total_balance[hour] * (self.electricity_prices[hour] / 1000)

        return electricity_cost
