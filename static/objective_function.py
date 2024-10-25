import numpy as np
import copy

class objective_function:
    """
    Objective function value of a magic cube.

    :var size: The dimensions of the Magic Cube
    :var magic_sum: The magic number/constant for the Magic Cube
    :var cube: A copy of magic cube from MagicCube class 
    """

    def __init__(self, magic_cube):
        """
        Generates the components of a magic cube.

        :param magic_cube: A copy of a magic cube class
        """
        self.size: int = magic_cube.size             
        self.magic_sum: int = magic_cube.magic_sum     
        self.cube = copy.deepcopy(magic_cube)

    def __str__(self):
        """
        Returns a string representation of magic cube's objective value.
        """
        return self.get_object_value().__str__()

    def get_object_value(self) -> int:
        """
        Returns objective value from internal function.

        :return: Objective function value of a magic cube
        """
        # Calculates objective function value
        return (109 - self.__check_315())
    
    def __check_315(self) -> int:
        """
        Returns the sum of a series (row/column/pillar/diagonal) from the cube that is not equal to the magic number.

        :return: The sum of wrong cube series (not equal to the magic number)
        """
        # Check all rows
        not_315_row = 0
        for z in range(self.size):
            for y in range(self.size):
                if np.sum(self.cube.get_row(y, z)) != self.magic_sum:
                    not_315_row += 1

        # Check all columns
        not_315_col = 0
        for z in range(self.size):
            for x in range(self.size):
                if np.sum(self.cube.get_col(x, z)) != self.magic_sum:
                    not_315_col += 1

        # Check all pillars
        not_315_pil = 0
        for y in range(self.size):
            for x in range(self.size):
                if np.sum(self.cube.get_pillar(x, y)) != self.magic_sum:
                   not_315_pil += 1

        # Check all space diagonals
        not_315_diag = 0
        for diag in self.cube.get_space_diags():
            if np.sum(diag) != self.magic_sum:
                not_315_diag += 1

        # Check all side diagonals on each axis
        not_315_diagx = 0
        for diag_x in self.cube.get_side_diags_x():
            if np.sum(diag_x) != self.magic_sum:
                not_315_diagx += 1

        not_315_diagy = 0
        for diag_y in self.cube.get_side_diags_y():
            if np.sum(diag_y) != self.magic_sum:
                not_315_diagy += 1

        not_315_diagz = 0
        for diag_z in self.cube.get_side_diags_z():
            if np.sum(diag_z) != self.magic_sum:
                not_315_diagz += 1

        return (not_315_row + not_315_col + not_315_pil + not_315_diag + not_315_diagx + not_315_diagy + not_315_diagz)
