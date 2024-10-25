from src.data_structure.magic_cube import MagicCube
import objective_function
import matplotlib.pyplot as plt
import copy
import time
import numpy as np

class hill_climb_steepest:
    """
    A local search algorithm: Hill-Climbing Steepest Ascent.

    :var initial: The initialization of a magic cube class depends on its size
    :var object_initial: The initialization of an objective function class
    """
    def __init__(self, size=5):
        """
        Generates an initial of Magic Cube and objective function class.

        :param size: Magic cube dimensions, default = 5
        """
        self.initial = MagicCube(size)
        self.object_initial = objective_function(self.initial)

    def plot_hc_steepest_ascent(self, object_values):
        """
        Returns a figure plot of objective function values.

        :param object_values: The array of objective function values
        :return: The plot of objective function
        """
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.plot(np.arange(len(object_values)), object_values)
        ax.set_title("HC Steepest Ascent Objective Function", fontsize=20, weight="bold")
        ax.set_xlabel("Iteration", fontsize=15)
        ax.set_ylabel("Objective Function Value", fontsize=15)
        ax.tick_params(labelsize=15)
        ax.grid()
        plt.close()
        
        return fig

    def hill_climb_steepest_ascent(self):
        """
        Returns the best state and objective function values.

        :return: The best state of magic cube, objective function values, number of iterations, and duration
        """
        time0 = time.time() 
        current = copy.deepcopy(self.initial)                   # copy of Magic cube class initial
        object_temp = self.object_initial.get_object_value()    # initial of objective function value
        object_values = [object_temp]                           # initiation of objective values array
        i = 0                                                   # initiation the number of iterations
        
        # Loop of hill-climbing steepest ascent
        while True: 
            # find the best state and its value
            neighbour, neighbour_value = self.__find_neigbour(current)   
            
            if neighbour_value <= object_temp:       
                # if the neighbour objective function value is LESS than or EQUAL to the current objective function value, stop the local search                               
                return current, np.array(object_values), i, (time.time() - time0)
            
            current = neighbour
            object_temp = neighbour_value
            object_values.append(object_temp)
            i += 1

    # -- INTERNAL FUNCTIONS --

    def __swap(self, cube, x1, x2):
        """
        Returns a swapped magic cube state.

        :param cube: A magic cube class
        :param x1: First target index
        :param x2: Second target index
        :return: A swapped magic cube state
        """
        cube_swap = copy.deepcopy(cube)
        cube_swap.data[[x1, x2]] = cube_swap.data[[x2, x1]]

        return cube_swap
    
    def __find_neigbour(self, cube):
        """
        Returns the best state and its objective function value.

        :param cube: A magic cube class
        :return: The best state and its objective function value
        """
        # Initialize arrays of successors and their objective values
        successors = []
        object_successors = []
        for x1 in range(len(cube.data)):
            for x2 in range(x1, len(cube.data)):
                if x1 != x2:  # Total iteration of all successor search: (125*124)/2 = 7750
                    cube_swap = self.__swap(cube, x1, x2)
                    object_value = objective_function(cube_swap)
                    successors.append(cube_swap)
                    object_successors.append(object_value.get_object_value())

        successors = np.array(successors)
        object_successors = np.array(object_successors)

        return successors[np.where(object_successors == object_successors.max())[0][0]], object_successors[np.where(object_successors == object_successors.max())[0][0]]
    