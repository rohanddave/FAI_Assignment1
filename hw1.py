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
    lambda1, lambda2, lambda3, lambda4 = 10000, 40, 35, 1000
    return (lambda1 * calculateConflictPenalty(grid) +
            lambda2 * calculateDistinctColors(grid) +
            lambda3 * len(shapes) + 
            lambda4 * getEmptyCellCount(grid))

def moveToCell(goalX, goalY):
    if 0 > goalX >= len(grid) or 0 > goalY >= len(grid[0]):
        return
    
    shapePos, _, _, _, _, _ = execute('export')

    currentY, currentX = shapePos

    while currentX != goalX:
        if goalX > currentX:
            currentX += 1
            execute('down')
        else:
            currentX -= 1
            execute('up')
    
    while currentY != goalY:
        if goalY > currentY:
            currentY += 1
            execute('right')
        else:
            currentY -= 1
            execute('left')

def countSurroundingEmptyCells(grid, m, n):
    empty_count = 0
    rows = len(grid)
    cols = len(grid[0])

    # Define the bounds of the 7x7 rectangle centered at (m, n)
    for i in range(m - 3, m + 4):  # 3 cells in each direction
        for j in range(n - 3, n + 4):  # 3 cells in each direction
            # Ensure we don't go out of bounds
            if 0 <= i < rows and 0 <= j < cols:
                if grid[i][j] == -1:
                    empty_count += 1

    return empty_count

def getRandomShape():
    shapePos, _, _, grid, _, _ = execute('export')
    emptyCells = countSurroundingEmptyCells(grid, shapePos[0], shapePos[1])
    totalCells = len(grid) * len(grid[0])

    # ones_count = [np.count_nonzero(shape == 1) for shape in shapes]
    shapeSize = [1, 2, 2, 4, 4, 4, 4, 3, 3]

    weights = [
    (abs(shapeSize[i] - emptyCells)) if (emptyCells >= shapeSize[i]) else 0 
    for i in range(len(shapes))
    ]

    total_weight = sum(weights)

    if total_weight == 0:
        return np.random.choice(len(shapes))
    
    probabilities = [w / total_weight for w in weights]

    return np.random.choice(len(shapes), p=probabilities)
# def getRandomShape():
#     return np.random.choice(len(shapes))

def switchToShape(goalShapeIndex):
    while True:
        _, currentShapeIndex, _, _, _, _ = execute('export')
        if currentShapeIndex == goalShapeIndex: 
            return 
        execute('switchshape')

def switchToColor(goalColorIndex):
    while True:
        _, _, currentColorIndex, _, _, _ = execute('export')
        if currentColorIndex == goalColorIndex: 
            return 
        execute('switchcolor')

def getRandomEmptyPosition():
    _, _, _, grid, _, _ = execute('export')

    # Find all empty cell indices
    indices = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == -1]

    # Check if there are any empty positions available
    if not indices:
        return None  # Return None if there are no empty cells
    
    # Randomly select one of the indices
    return random.choice(indices)

def getRandomNeighbor(s):
    attempts = 0
    max_attempts = 10

    while attempts < max_attempts:
        i, j = getRandomEmptyPosition()
        # move to position 
        moveToCell(i, j)

        # get current grid and bursh position
        shapePos, _, _, grid, _, _ = execute('export')

        randomColorIndex = getAvailableColor(grid, shapePos[1], shapePos[0])
        randomShapeIndex = 0 if attempts == max_attempts - 1 else getRandomShape()
        
        # switch to random shape in env
        switchToShape(randomShapeIndex)

        # switching to random color in env
        switchToColor(randomColorIndex)

        if canPlace(grid, shapes[randomShapeIndex], shapePos):
            execute('place')
            return execute('export')
        else:
            attempts += 1
    return s

def hillClimbing(s):
    current = s

    while True:
        # Get the current state and calculate its value
        _, _, _, grid, placedShapes, _ = current
        current_value = calculateObjectiveFunction(grid, placedShapes)

        if getEmptyCellCount(grid) == 0:
            return current
        
        # print("trying to get a random neighbor")
        neighbor = getRandomNeighbor(current)
        
        _, _, _, neighbor_grid, neighbor_placed_shapes, _ = neighbor

        if getEmptyCellCount(neighbor_grid) == 0: 
            return neighbor
        
        neighbor_value = calculateObjectiveFunction(neighbor_grid, neighbor_placed_shapes)

        if neighbor_value < current_value:
            current = execute('export')
        else:
            execute('undo')
            current = execute('export')  


s = hillClimbing(execute('export'))
_, _, _, solutionGrid, solutionPlacedShapes, _ = s
print("FOUND SOLUTION!!!!!")
print(calculateObjectiveFunction(solutionGrid, solutionPlacedShapes))
########################################

# Do not modify any of the code below. 

########################################

end=time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))
