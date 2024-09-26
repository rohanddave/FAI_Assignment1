import time
import numpy as np
from gridgame import *

##############################################################################################################################

# You can visualize what your code is doing by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.

# For your final submission, please set the GUI option to False.

# The gs argument controls the grid size. You should experiment with various sizes to ensure your code generalizes.

##############################################################################################################################

setup(GUI = True, render_delay_sec = 0.001, gs = 10)


##############################################################################################################################

# Initialization

# shapePos is the current position of the brush.

# currentShapeIndex is the index of the current brush type being placed (order specified in gridgame.py, and assignment instructions).

# currentColorIndex is the index of the current color being placed (order specified in gridgame.py, and assignment instructions).

# grid represents the current state of the board. 
    
    # -1 indicates an empty cell
    # 0 indicates a cell colored in the first color (indigo by default)
    # 1 indicates a cell colored in the second color (taupe by default)
    # 2 indicates a cell colored in the third color (veridian by default)
    # 3 indicates a cell colored in the fourth color (peach by default)

# placedShapes is a list of shapes that have currently been placed on the board.
    
    # Each shape is represented as a list containing three elements: a) the brush type (number between 0-8), 
    # b) the location of the shape (coordinates of top-left cell of the shape) and c) color of the shape (number between 0-3)

    # For instance [0, (0,0), 2] represents a shape spanning a single cell in the color 2=veridian, placed at the top left cell in the grid.

# done is a Boolean that represents whether coloring constraints are satisfied. Updated by the gridgames.py file.

##############################################################################################################################

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = execute('export')

# input()   # <-- workaround to prevent PyGame window from closing after execute() is called, for when GUI set to True. Uncomment to enable.
print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)


####################################################
# Timing your code's execution for the leaderboard.
####################################################

start = time.time()  # <- do not modify this.



##########################################
# Write all your code in the area below. 
##########################################


'''

YOUR CODE HERE


'''

# Objective function helpers
def calculateConflictPenalty(grid):
    rows = len(grid)
    cols = len(grid[0])
    conflicts = 0

    for i in range(rows):
        for j in range(cols):
            current_cell = grid[i][j]

            if current_cell == -1:
                continue

            # Check for adjacent conflicts (4-neighborhood)
            neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for neighbor in neighbors:
                x, y = i + neighbor[0], j + neighbor[1]
                if 0 <= x < rows and 0 <= y < cols and grid[x][y] == current_cell:
                    conflicts += 1
    
    return conflicts // 2  # Each conflict is counted twice, so divide by 2

def calculateDistinctColors(grid):
    seen_colors = set()
    for row in grid:
        for cell in row:
            if cell != -1:  # Exclude empty cells
                seen_colors.add(cell)
    return len(seen_colors)

def getEmptyCellCount(grid): 
    count = 0

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == -1:
                count += 1

    return count

# objective function
def calculateObjectiveFunction(grid, shapes):
    # Coefficients for penalties (adjust if needed)
    lambda1, lambda2, lambda3, lambda4 = 3, 2, 2, 5
    return (lambda1 * calculateConflictPenalty(grid) +
            lambda2 * calculateDistinctColors(grid) +
            lambda3 * len(shapes) + 
            lambda4 * getEmptyCellCount(grid))

# Utility functions for getting neighbors 
# Random shape selector with probability weights
# TODO: if cannot place shape then ideally we should reduce the weight of all the ones larger than the one rejected now 
def getRandomShape():
    return np.random.choice(len(shapes))
    # weights = np.arange(1, len(shapes) + 1)
    # probabilities = weights / weights.sum()
    # return np.random.choice(len(shapes), p=probabilities)

def switchToColor(goalColorIndex):
    print('switch to color called with: ' + str(goalColorIndex))
    # NOTE: this has no check if the index is bounded hence can be an infinite loop
    while True:
        _, _, currentColorIndex, _, _, _ = execute('export')
        if currentColorIndex == goalColorIndex: 
            return 
        execute('switchcolor')
    print('reached goal color')

# switch to shape in env
def switchToShape(goalShapeIndex):
    print('switch to shape called with: ' + str(goalShapeIndex))
    while True:
        _, currentShapeIndex, _, _, _, _ = execute('export')
        if currentShapeIndex == goalShapeIndex: 
            return 
        execute('switchshape')
    print('reached goal shape')

# Move to a specific cell in the grid
def moveToCell(goalX, goalY):
    shapePos, _, _, _, _, _ = execute('export')
    if shapePos[0] == goalX and shapePos[1] == goalY:
        return
    
    hasReachedX = shapePos[0] != goalX
    hasReachedY = shapePos[1] != goalY
    while not hasReachedX or not hasReachedY:
        shapePos, _, _, _, _, _ = execute('export')
        hasReachedX = shapePos[0] != goalX
        hasReachedY = shapePos[1] != goalY
        # If we're already at the target position, stop
        if shapePos[0] == goalX and shapePos[1] == goalY:
            return
        if shapePos[0] != goalX:
            vertical_command = 'down' if goalX > shapePos[0] else 'up'
            execute(vertical_command)
        if shapePos[1] != goalY:
            horizontal_command = 'right' if goalY > shapePos[1] else 'left'
            execute(horizontal_command)

def getRandomEmptyPosition(array):
    indices = [(i, j) for i in range(len(array)) for j in range(len(array[i])) if array[i][j] == -1]
    # Randomly select one of the indices
    return random.choice(indices)

# get neighbors function
def getRandomNeighbor(s):
    _, _, _, grid, _, _ = s
    
    while True:
        i, j = getRandomEmptyPosition(grid)
        # move to position 
        moveToCell(i, j)

        # get current grid and bursh position
        shapePos, _, _, grid, _, _ = execute('export')

        randomColorIndex = getAvailableColor(grid, shapePos[0], shapePos[1])
        randomShapeIndex = getRandomShape()

        # switch to random shape in env
        switchToShape(randomShapeIndex)

        # switching to random color in env
        switchToColor(randomColorIndex)

        if canPlace(grid, shapes[randomShapeIndex], shapePos):
            execute('place')
            return execute('export')

# hill climbing algorithm
def hillClimbing(s):
    current = s

    while True:
        # Get the current state and calculate its value
        _, _, _, grid, placedShapes, currentDone = current
        current_value = calculateObjectiveFunction(grid, placedShapes)

        if currentDone or checkGrid(grid):
            return current
        
        neighbor = getRandomNeighbor(current)
        _, _, _, neighbor_grid, neighbor_placed_shapes, done = neighbor

        if done: 
            return neighbor
        
        neighbor_value = calculateObjectiveFunction(neighbor_grid, neighbor_placed_shapes)

        if neighbor_value < current_value:
            current = neighbor
        else:
            execute('undo')
            current = execute('export')  

# Run hill climbing starting from the initial state
hillClimbing(execute('export'))

########################################

# Do not modify any of the code below. 

########################################

end=time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))
