import time
import numpy as np
from gridgame import *

##############################################################################################################################

# You can visualize what your code is doing by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.

# For your final submission, please set the GUI option to False.

# The gs argument controls the grid size. You should experiment with various sizes to ensure your code generalizes.

##############################################################################################################################

gridSize = 5
setup(GUI = True, render_delay_sec = 0.001, gs = gridSize)


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
    lambda1, lambda2, lambda3, lambda4 = 50, 8, 2, 1000
    return (lambda1 * calculateConflictPenalty(grid) +
            lambda2 * calculateDistinctColors(grid) +
            lambda3 * len(shapes) + 
            lambda4 * getEmptyCellCount(grid))

# Utility functions for getting neighbors 
# Random shape selector with probability weights
def getRandomShape():
    """
    Distributes probability more towards the last elements when there are more empty cells,
    and shifts the probability towards the starting elements as the number of empty cells decreases.
    
    :param elements: A list of elements to choose from.
    :param emptyCells: The current number of empty cells remaining.
    :return: A randomly chosen element with skewed probability.
    """
    elements = [i for i in range(len(shapes))]

    _, _, _, grid, _, _ = execute('export')
    emptyCells = getEmptyCellCount(grid)
    # Total number of elements
    n = len(elements)
    
    # Calculate the ratio of empty cells to total elements
    empty_ratio = emptyCells / pow(len(grid), 2)
    print("empty ratio" + str(empty_ratio))
    
    # Skew factor: bias towards later elements when empty_ratio is high, and earlier elements when low
    weights = [(empty_ratio ** i) if empty_ratio > 0.5 else ((1 - empty_ratio) ** i) for i in range(n)]
    
    # Normalize the weights to sum to 1 (to form a valid probability distribution)
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    print('probabilities')
    print(probabilities)
    
    # Select an element based on the weighted probabilities
    chosen_element = random.choices(elements, probabilities)[0]
    
    return chosen_element

# def getRandomShape():
#     return np.random.choice(len(shapes))
#     # weights = np.arange(1, len(shapes) + 1)
#     # probabilities = weights / weights.sum()
#     # return np.random.choice(len(shapes), p=probabilities)

def switchToColor(goalColorIndex):
    # print('switch to color called with: ' + str(goalColorIndex))
    # NOTE: this has no check if the index is bounded hence can be an infinite loop
    while True:
        _, _, currentColorIndex, _, _, _ = execute('export')
        if currentColorIndex == goalColorIndex: 
            print('reached goal color') 
            return 
        execute('switchcolor')

# switch to shape in env
def switchToShape(goalShapeIndex):
    # print('switch to shape called with: ' + str(goalShapeIndex))
    while True:
        _, currentShapeIndex, _, _, _, _ = execute('export')
        if currentShapeIndex == goalShapeIndex: 
            print('reached goal shape')
            return 
        execute('switchshape')

# Move to a specific cell in the grid
def moveToCell(goalX, goalY):
    # print("Attempting to move to: " + str((goalX, goalY)))
    
    # Ensure goal is within grid bounds
    if not (0 <= goalX < gridSize and 0 <= goalY < gridSize):
        print("Error: Goal is outside the grid boundaries.")
        return

    while True:
        shapePos, _, _, _, _, _ = execute('export')
        # print("Current position: " + str(shapePos))

        # If we're already at the target position, stop
        if shapePos[0] == goalX and shapePos[1] == goalY:
            # print('Moved to x: ' + str(goalX) + '\t y: ' + str(goalY))
            return

        # Track position before moving to avoid getting stuck
        prevPos = shapePos
        
        # Move along the X axis
        if goalX > shapePos[0]:
            execute('down')
        elif goalX < shapePos[0]:
            execute('up')
        
        # Move along the Y axis
        if goalY > shapePos[1]:
            execute('right')
        elif goalY < shapePos[1]:
            execute('left')
        
        # Re-fetch position after move to ensure the shape actually moved
        shapePos, _, _, _, _, _ = execute('export')
        
        # If the shape didn't move, avoid infinite loops
        if shapePos == prevPos:
            print("Warning: No movement detected. Stopping.")
            return

def getRandomEmptyPosition():
    _, _, _, grid, _, _ = execute('export')

    # Find all empty cell indices
    indices = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == -1]

    # Check if there are any empty positions available
    if not indices:
        return None  # Return None if there are no empty cells
    
    # Randomly select one of the indices
    return random.choice(indices)

# get neighbors function
def getRandomNeighbor(s):
    while True:
        i, j = getRandomEmptyPosition()
        # print('got random position')
        # move to position 
        moveToCell(i, j)

        # get current grid and bursh position
        shapePos, _, _, grid, _, _ = execute('export')
        # print('got current state')

        randomColorIndex = getAvailableColor(grid, shapePos[0], shapePos[1])
        randomShapeIndex = getRandomShape()
        # print('random color: ' + str(randomColorIndex))
        # print('random shape index: ' + str(randomShapeIndex))
        # switch to random shape in env
        switchToShape(randomShapeIndex)

        # switching to random color in env
        switchToColor(randomColorIndex)

        if canPlace(grid, shapes[randomShapeIndex], shapePos):
            execute('place')
            return execute('export')
        else:
            print('cannot place this neighbor')
# def getRandomNeighbor(s):
#     _, _, _, grid, _, _ = s
#     attempts = 0 
#     maxAttempts = 100
    
#     while attempts <= maxAttempts:
#         print('while loop start for attemp number: ' + str(attempts))
#         i, j = getRandomEmptyPosition(grid)
#         print('got random position')
#         # move to position 
#         moveToCell(i, j)

#         # get current grid and bursh position
#         shapePos, _, _, grid, _, _ = execute('export')
#         print('got current state')

#         randomColorIndex = getAvailableColor(grid, shapePos[0], shapePos[1])
#         randomShapeIndex = getRandomShape()
#         print('random color: ' + str(randomColorIndex))
#         print('random shape index: ' + str(randomShapeIndex))
#         # switch to random shape in env
#         switchToShape(randomShapeIndex)

#         # switching to random color in env
#         switchToColor(randomColorIndex)

#         if canPlace(grid, shapes[randomShapeIndex], shapePos):
#             execute('place')
#             return execute('export')
#         else:
#             print('cannot place this neighbor')
#         attempts += 1
#     print("could not find a neighbor")
            

# hill climbing algorithm
def hillClimbing(s):
    current = s

    while True:
        # Get the current state and calculate its value
        _, _, _, grid, placedShapes, currentDone = current
        current_value = calculateObjectiveFunction(grid, placedShapes)

        if currentDone or checkGrid(grid):
            return current
        
        # print("trying to get a random neighbor")
        neighbor = getRandomNeighbor(current)
        # print('got a random neighbor')
        _, _, _, neighbor_grid, neighbor_placed_shapes, done = neighbor

        if done or checkGrid(neighbor_grid): 
            print('neighbor is solution')
            return neighbor
        
        neighbor_value = calculateObjectiveFunction(neighbor_grid, neighbor_placed_shapes)

        if neighbor_value < current_value:
            print('neighbor improves objective function: accept as solution')
            current = neighbor
        else:
            print('neighbor worsens objective function: backtrack')
            execute('undo')
            current = execute('export')  

# def hillClimbingWithRestarts(s, max_restarts=10, max_no_improve=100):
#     current = s
#     restart_count = 0
#     no_improvement = 0

#     while restart_count < max_restarts:
#         # Get the current state and calculate its value
#         _, _, _, grid, placedShapes, currentDone = current
#         current_value = calculateObjectiveFunction(grid, placedShapes)

#         if currentDone or checkGrid(grid):
#             return current
        
#         print("trying to get a random neighbor")
#         neighbor = getRandomNeighbor(current)
#         print('got a random neighbor')
#         _, _, _, neighbor_grid, neighbor_placed_shapes, done = neighbor

#         if done or checkGrid(neighbor_grid): 
#             print('neighbor is solution')
#             return neighbor
        
#         neighbor_value = calculateObjectiveFunction(neighbor_grid, neighbor_placed_shapes)

#         if neighbor_value < current_value:
#             print('neighbor improves objective function: accept as solution')
#             current = neighbor
#             no_improvement = 0  # Reset no improvement counter
#         else:
#             print('neighbor worsens objective function: backtrack')
#             execute('undo')
#             current = execute('export')
#             no_improvement += 1

#         # If no improvement for too long, restart the algorithm
#         if no_improvement >= max_no_improve:
#             print(f'Restarting after {no_improvement} iterations without improvement.')
#             current = randomRestart()  # Restart with a new random state
#             no_improvement = 0
#             restart_count += 1

#     return current

# def randomRestart():
#     # Restart with a random valid initial state
#     print("Random restart triggered!")
#     # Execute the initial state or reset parts of the grid as needed
#     execute('reset')
#     return execute('export')

# Run hill climbing with restarts
# hillClimbingWithRestarts(execute('export'))

# Run hill climbing starting from the initial state
hillClimbing(execute('export'))
print('completed hill climbing')

########################################

# Do not modify any of the code below. 

########################################

end=time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))
