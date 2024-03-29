import numpy as np
import math
from .optimizer import Optimizer
from .utils import _correct_bounds, _scale_with_threshold, _tpa_mutation
from .parameters import Parameters
from .population import Population


class ConfigurableCMAES(Optimizer):
    '''The main class of the configurable CMA ES continous optimizer.

    Attributes
    ----------
    _fitness_func: callable
        The objective function to be optimized
    parameters: Parameters
        All the parameters of the CMA ES algorithm are stored in
        the parameters object. Note if a parameters object is not
        explicitly passed, all *args and **kwargs passed into the
        constructor of a ConfigurableCMAES are directly passed into
        the constructor of a Parameters object.
    '''

    def __init__(
            self,
            fitness_func,
            *args,
            parameters=None,
            **kwargs
    ) -> None:
        self._fitness_func = fitness_func
        self.parameters = parameters if isinstance(
            parameters, Parameters
        ) else Parameters(*args, **kwargs)
        #self.initial_distances = None
        #self.sigma_vec = []
        self.eval_count = []
        self.stepsizes = []
        self.matrix_norm = []
        self.C_overtime = []
        self.x_hist = []
        self.a2_x_hist = []
        self.f_hist = []
        self.prec_overtime = []
        #self.ps_overtime = []
        #self.pc_overtime = []

    def get_mutation_generator(self) -> None:
        '''Returns a mutation generator, which performs mutation.
        First, a directional vector zi is sampled from a sampler object
        as defined in the self.parameters object. Then, this zi vector is
        multiplied with the eigenvalues D, and the dot product is taken with the
        eigenvectors B of the covariance matrix C in order to create a scaled
        directional mutation vector yi. By scaling this vector with current population
        mean m, and the step size sigma, a new individual xi is created. The
        self.fitness_func is called (where?) in order to compute the fitness of the newly created
        individuals.

        If the step size adaptation method is 'tpa', two less 'normal'
        individuals are created.

        '''
        y, x, f = [], [], []
        n_offspring = self.parameters.lambda_
        if self.parameters.step_size_adaptation == 'tpa' and self.parameters.old_population:
            n_offspring -= 2
            _tpa_mutation(self.fitness_func, self.parameters, x, y, f)

        for i in range(1, n_offspring + 1):
            zi = next(self.parameters.sampler)
            if self.parameters.threshold_convergence:
                zi = _scale_with_threshold(zi, self.parameters.threshold)

            # B is a matrix of C's eigenvectors, D is an array with C's eigenvalues
            #print(f'B: {self.parameters.B} D: {self.parameters.D}')
            yi = np.dot(self.parameters.B, self.parameters.D * zi) # dot product of matrix B and element-wise multiplied D * zi
            xi = self.parameters.m + (self.parameters.sigma * yi) # formula xi = m + sigma * Ni(0,C) where yi  = Ni(0, C)
            if self.parameters.bound_correction:
                xi = _correct_bounds(
                    xi, self.parameters.ub, self.parameters.lb)

            fi = yield yi, xi # why can you evaluate the fitness like this?
            [a.append(v) for a, v in ((y, yi), (x, xi), (f, fi),)]

            if self.sequential_break_conditions(i, fi):
                break

        self.parameters.population = Population(
            np.hstack(x),
            np.hstack(y),
            np.array(f))

    def mutate(self):
        '''Method performing mutation and evaluation of a set of individuals.
        Collects the output of the mutation generator.
        '''
        mutation_generator = self.get_mutation_generator()
        yi, xi = next(mutation_generator)
        while True:
            try:
                yi, xi = mutation_generator.send(self.fitness_func(xi))
            except StopIteration:
                break

    def select(self) -> None:
        '''Selection of best individuals in the population
        The population is sorted according to their respective fitness
        values. Normally, the mu best individuals would be selected afterwards.
        However, because the option of active update is available, (and we could
        potentially need the mu worst individuals) the lambda best indivduals are
        selected. In recombination, only the mu best individuals are used to recompute
        the mean, so implicited selection happens there.

        If elistism is selected as an option, the mu best individuals of the old
        population are added to the pool of indivduals before sorting.

        If selection is to be performed pairwise, the only the best individuals
        of sequential pairs are used, the others are discarded. The intended
        use for this functionality is with mirrored sampling, in order to counter the
        bias generated by this sampling method. This method cannot be performed when there
        is an odd number of individuals in the population.
        '''
        if self.parameters.mirrored == 'mirrored pairwise':
            if not len(self.parameters.population.f) % 2 == 0:
                raise ValueError(
                        'Cannot perform pairwise selection with '
                        'an odd number of indivuduals'
                )
            indices = [int(np.argmin(x) + (i * 2))
                       for i, x in enumerate(np.split(
                           self.parameters.population.f,
                           len(self.parameters.population.f) // 2))
                       ]
            self.parameters.population = self.parameters.population[indices]

        if self.parameters.elitist and self.parameters.old_population:
            self.parameters.population += self.parameters.old_population[
                : self.parameters.mu]

        self.parameters.population.sort()

        self.parameters.population = self.parameters.population[
            : self.parameters.lambda_]

        self.parameters.fopt = min(
            self.parameters.fopt, self.parameters.population.f[0])

    def recombine(self) -> None:
        '''Recombination of new individuals
        In the CMAES, recombination is not as explicit as in for example
        a genetic algorithm. In the CMAES, recombination happens though the
        moving of the mean m, by multiplying the old mean with a weighted
        combination of the current mu best individuals.
        '''

        self.parameters.m_old = self.parameters.m.copy()
        self.parameters.m = self.parameters.m_old + (1 * (
            (self.parameters.population.x[:, :self.parameters.mu] -
                self.parameters.m_old) @
            self.parameters.pweights).reshape(-1, 1)
        )

            
    def step(self) -> bool:
        '''The step method runs one iteration of the optimization process.
        The method is called within the self.run loop, as defined in the
        Optimizer parent class. In there, a while loop runs until this step
        function returns a Falsy value.

        Returns
        -------
        bool
            Denoting whether to keep running this step function.
        '''

        
        #Append information to create plots
        self.stepsizes.append(self.parameters.sigma)   
        self.matrix_norm.append(np.linalg.norm(self.parameters.C))
        self.eval_count.append(self._fitness_func.evaluations)
        self.C_overtime.append(self.parameters.C)
        
        self.mutate()

        """Validate pop initialization in the neighbourhood of x_opt"""
        """if self.initial_distances is None:
            print(f' search space size: {math.sqrt(100 * self.parameters.d)}')
            self.initial_distances = []
            for i in range(0, self.parameters.lambda_):
                a = self.parameters.population.x[:, i]
                self.initial_distances.append(np.linalg.norm(a - self.parameters.m.flatten()))
                print(f' init_distance: {self.initial_distances[i]}')
                
        """

        self.select()
        self.recombine()
        self.parameters.adapt()
        
        for i in range(0, self.parameters.lambda_):
            self.x_hist.append(self.parameters.population.x[:, i])
            self.a2_x_hist.append(self.parameters.population.x[:, i])
            self.f_hist.append(self.parameters.population.f[i])

        self.prec_overtime.append(self._fitness_func.best_so_far_precision)

        return not any(np.atleast_1d(self.break_conditions()))
        

    def sequential_break_conditions(self, i: int, f: float) -> bool:
        '''Method returning a boolean value, indicating whether there are any
        sequential break conditions.

        Parameters
        ----------
        i: int
            The number of individuals already generated this current
            generation.
        f: float
            The fitness of that individual

        Returns
        -------
        bool
        '''
        if self.parameters.sequential:
            return (f < self.parameters.fopt and
                    i >= self.parameters.seq_cutoff and (
                        self.parameters.mirrored != 'mirrored pairwise'
                        or i % 2 == 0
                        )
                    )
        return False
