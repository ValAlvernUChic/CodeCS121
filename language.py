"""
CS 121: Language shifts

VAL ALVERN CUECO LIGO

Functions for language shift simulation.

This program takes the following parameters:
    grid _file (string): the name of a file containing a sample region
    R (int): neighborhood radius
    A (float): the language state transition threshold A
    Bs (list of floats): a list of the transition thresholds B to
      use in the simulation
    C (float): the language state transition threshold C
      centers (list of tuples): a list of community centers in the
      region
    max_steps (int): maximum number of steps

Example use:
    $ python3 language.py -grid_file tests/writeup-grid-with-cc.txt
	  --r 2 --a 0.5 --b 0.9 --c 1.2 --max_steps 5
While shown on two lines, the above should be entered as a single command.
"""

import copy
import click
import utility


def adjacent(grid, dist, loc):
    """
    To find adjacent grids (including loc) according to number of steps.

    Input:
      grid (list of lists of ints): the grid
      dist (int): distance from loc
      loc (tuple): tuple of grid location

    Returns: tuples of adjacent grids (tuple)
    """
    row, col = loc
    for x in range(-dist, dist + 1):
        for y in range(-dist, dist + 1):
            if 0 <= (row + x) < len(grid) and 0 <= (col + y) < len(grid):
                neighbor = (row + x, col + y)
                yield neighbor

def near_sc(grid, centers, loc):
    """
    Determine if an SL-speaking home is within the service distance of a CC.

    Inputs:
      grid (list of lists of ints): the grid
      centers (list of tuples): a list of community centers in the region
      loc (tuple): tuple of grid location

    Returns: bool value of whether loc is within sc (bool)
    """
    r, c = loc
    for x in centers:
        center, prox = x
        gr = adjacent(grid, prox, center)
        for y in gr:
            serve = True
            if loc != y or grid[r][c] == 0:
                serve = False
            if serve:
                return serve

    if centers == []:
        serve = False

    return serve

def engagement_level(grid, R, loc):
    """
    Calculate the engagement level of individual grids.

    Inputs:
      grid (list of lists of ints): the grid
      R (int): neighborhood radius
      loc (tuple): tuple of grid location

    Returns: engagement level (int)
    """
    s = 0
    t = sum(1 for x in adjacent(grid, R, loc))
    e = 0
    for state in adjacent(grid, R, loc):
        r, c = state
        s += grid[r][c]
        e = s / t

    return e

def language_state(grid, R, thresholds, centers, loc):
    """
    Calculate new language state of grid.

    Inputs:
      grid (list of lists of ints): the grid
      R (int): neighborhood radius
      thresholds (float, float, float): the language state
                                      transition thresholds (A, B, C)
      centers (list of tuples): a list of community centers in the region
      loc (tuple): tuple of grid location

    Returns: new language state of grid (int)
    """
    A, B, C = thresholds
    state = 0
    E = engagement_level(grid, R, loc)
    r, c = loc
    home = grid[r][c]
    if home == 0:
        if E <= B:
            state = 0
        else:
            state = 1

    elif home == 1:
        if E < B and not near_sc(grid, centers, loc):
            state = 0
        elif E <= C:
            state = 1
        else:
            state = 2

    elif home == 2:
        if E <= A and not near_sc(grid, centers, loc):
            state = 0
        elif E < B and not near_sc(grid, centers, loc):
            state = 1
        else:
            state = 2

    return state

def simulate_step(grid, R, thresholds, centers):
    """
    Simulates a single generational step in language state.

    Inputs:
      grid (list of lists of ints): the grid
      R (int): neighborhood radius
      thresholds (float, float, float): the language
        state transition thresholds (A, B, C)
      centers (list of tuples): a list of community centers in the
        region

    Returns: modified grid of new language states (list of lists)
    """
    changes = 0
    zero, one , two = 0, 0, 0
    for r, _ in enumerate(grid):
        for c, _ in enumerate(grid):
            original = grid[r][c]
            grid[r][c] = language_state(grid, R, thresholds, centers, (r, c))
            if grid[r][c] != original:
                changes += 1
            if grid[r][c] == 0:
                zero += 1
            elif grid[r][c] == 1:
                one += 1
            elif grid[r][c] == 2:
                two += 1

    return changes, zero, one, two

def run_simulation(grid, R, thresholds, centers, max_steps):
    """
    Do the simulation.

    Inputs:
      grid (list of lists of ints): the grid
      R (int): neighborhood radius
      thresholds (float, float, float): the language
        state transition thresholds (A, B, C)
      centers (list of tuples): a list of community centers in the
        region
      max_steps (int): maximum number of steps

    Returns: the frequency of each language state (int, int, int)
    """
    steps = 0
    for steps in range(max_steps):
        steps += 1
        changes, zero, one, two = \
            simulate_step(grid, R, thresholds, centers)
        if changes == 0:
            break

    return (zero, one, two)

def simulation_sweep(grid, R, A, Bs, C, centers, max_steps):
    """
    Run the simulation with various values of threshold B.

    Inputs:
      grid (list of lists of ints): the grid
      R (int): neighborhood radius
      A (float): the language state transition threshold A
      Bs (list of floats): a list of the transition thresholds B to
        use in the simulation
      C (float): the language state transition threshold C
      centers (list of tuples): a list of community centers in the
        region
      max_steps (int): maximum number of steps

    Returns: a list of frequencies (tuples) of language states for
      each threshold B.
    """
    freqs = []
    for x in Bs:
        g = copy.deepcopy(grid)
        thresholds = (A, x, C)
        freq = run_simulation(g, R, thresholds, centers, max_steps)
        freqs.append(freq)

    return freqs

@click.command(name="language")
@click.option('--grid_file', type=click.Path(exists=True),
              default="tests/writeup-grid.txt", help="filename of the grid")
@click.option('--r', type=int, default=1, help="neighborhood radius")
@click.option('--a', type=float, default=0.6, help="transition threshold A")
@click.option('--b', type=float, default=0.8, help="transition threshold B")
@click.option('--c', type=float, default=1.6, help="transition threshold C")
@click.option('--max_steps', type=int, default=1,
              help="maximum number of simulation steps")
def cmd(grid_file, r, a, b, c, max_steps):
    """
    Run the simulation.
    """

    grid, centers = utility.read_grid(grid_file)
    print_grid = len(grid) < 20

    print("Running the simulation...")

    if print_grid:
        print("Initial region:")
        for row in grid:
            print("   ", row)
        if len(centers) > 0:
            print("With community centers:")
            for center in centers:
                print("   ", center)

    # run the simulation
    frequencies = run_simulation(grid, r, (a, b, c), centers, max_steps)

    if print_grid:
        print("Final region:")
        for row in grid:
            print("   ", row)

    print("Final language state frequencies:", frequencies)


if __name__ == "__main__":
    cmd()
