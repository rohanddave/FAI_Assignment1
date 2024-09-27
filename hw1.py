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
    lambda1, lambda2, lambda3, lambda4 = 50, 20, 8, 1000
    return (lambda1 * calculateConflictPenalty(grid) +
            lambda2 * calculateDistinctColors(grid) +
            lambda3 * len(shapes) + 
            lambda4 * getEmptyCellCount(grid))

# Utility functions for getting neighbors 
# Random shape selector with probability weights
# def getRandomShape():
#     """
#     Distributes probability more towards the last elements when there are more empty cells,
#     and shifts the probability towards the starting elements as the number of empty cells decreases.
    
#     :param elements: A list of elements to choose from.
#     :param emptyCells: The current number of empty cells remaining.
#     :return: A randomly chosen element with skewed probability.
#     """
#     elements = [i for i in range(len(shapes))]

#     _, _, _, grid, _, _ = execute('export')
#     emptyCells = getEmptyCellCount(grid)
#     # Total number of elements
#     n = len(elements)
    
#     # Calculate the ratio of empty cells to total elements
#     empty_ratio = emptyCells / pow(len(grid), 2)
#     # print("empty ratio" + str(empty_ratio))
    
#     # Skew factor: bias towards later elements when empty_ratio is high, and earlier elements when low
#     weights = [(empty_ratio ** i) if empty_ratio > 0.5 else ((1 - empty_ratio) ** i) for i in range(n)]
    
#     # Normalize the weights to sum to 1 (to form a valid probability distribution)
#     total_weight = sum(weights)
#     probabilities = [w / total_weight for w in weights]
#     # print('probabilities')
#     # print(probabilities)
    
#     # Select an element based on the weighted probabilities
#     chosen_element = random.choices(elements, probabilities)[0]
    
#     return chosen_element

# def getRandomShape():
#     """
#     Distributes probability more towards the last elements when there are more empty cells,
#     and shifts the probability towards the starting elements as the number of empty cells decreases.
    
#     :return: A randomly chosen element with skewed probability.
#     """
#     elements = [i for i in range(len(shapes))]

#     # Get grid and empty cells count
#     _, _, _, grid, _, _ = execute('export')
#     emptyCells = getEmptyCellCount(grid)
    
#     # Total number of elements
#     n = len(elements)
    
#     # Calculate the ratio of empty cells to the total number of cells in the grid
#     total_cells = pow(len(grid), 2)
#     empty_ratio = emptyCells / total_cells
#     print('empty ratio: ' + str(empty_ratio))
    
#     # Adjust weights to favor first edge when fewer empty cells, last edge when more empty cells
#     if empty_ratio > 0.5:
#         # More empty cells -> favor last elements
#         weights = [(empty_ratio ** i) for i in range(n)]
#     else:
#         # Fewer empty cells -> favor first elements
#         weights = [((1 - empty_ratio) ** (n - i - 1)) for i in range(n)]
    
#     print(weights)
#     # Normalize the weights to sum to 1 (to form a valid probability distribution)
#     total_weight = sum(weights)
#     probabilities = [w / total_weight for w in weights]
#     print(probabilities)
    
#     # Select an element based on the weighted probabilities
#     chosen_element = random.choices(elements, probabilities)[0]
    
#     return chosen_element

# def getRandomShape():
#     """
#     Adjusts the peak of the probability distribution curve along the length of the elements
#     based on the empty ratio. The peak moves towards the last elements when there are more empty cells,
#     and shifts towards the first elements as the number of empty cells decreases.
    
#     :return: A randomly chosen element with skewed probability.
#     """
#     elements = [i for i in range(len(shapes))]

#     # Get grid and empty cells count
#     _, _, _, grid, _, _ = execute('export')
#     emptyCells = getEmptyCellCount(grid)
    
#     # Total number of elements
#     n = len(elements)
    
#     # Calculate the ratio of empty cells to the total number of cells in the grid
#     total_cells = pow(len(grid), 2)
#     empty_ratio = emptyCells / total_cells
#     print('empty ratio: ' + str(empty_ratio))
    
#     # Determine the peak location: closer to the first elements if fewer empty cells, and closer to the last elements if more empty cells
#     peak = int(empty_ratio * (n - 1))  # Peak moves from 0 to n-1 depending on empty_ratio
    
#     # Spread factor: controls the width of the probability distribution
#     spread = max(1, int(n * (0.5)))  # A constant spread (can be adjusted)
    
#     # Gaussian-like weight calculation centered at the 'peak'
#     def gaussian(x, mu, sigma):
#         return math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
#     # Calculate weights based on the Gaussian distribution centered at the peak
#     weights = [gaussian(i, peak, spread) for i in range(n)]
    
#     print("Weights before normalization:", weights)
    
# def getRandomShape():
#     shapePos, _, _, grid, _, _ = execute('export')
#     emptyCells = countSurroundingEmptyCells(grid, shapePos[0], shapePos[1])
#     totalCells = len(grid) * len(grid[0])
    
#     # ones_count = [np.count_nonzero(shape == 1) for shape in shapes]
#     shapeSize = [1, 2, 2, 4, 4, 4, 4, 3, 3]
        
#     weights = [
#     (abs(shapeSize[i] - emptyCells)) if (emptyCells >= shapeSize[i]) else 0 
#     for i in range(len(shapes))
#     ]
#     total_weight = sum(weights)
#     probabilities = [w / (total_weight if total_weight != 0 else 1) for w in weights]
#     print(probabilities)
#     return np.random.choice(len(shapes), p=probabilities)

# def getBestFitShape():
#     current = execute('export')
#     best_state = current
#     shapePos, _, _, grid, _, _  = current
#     best_value = calculateObjectiveFunction(grid, shapes)
#     for i in range(len(shapes)):
#         switchToShape(i)
#         execute('place')
#         new_state = execute('export')
#         shapePos, _, _, newGrid, _, _ = new_state
#         new_value = calculateObjectiveFunction(newGrid, shapes)
#         if new_value < best_value:
#             best_state = new_state
#             best_value = new_value
        

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

def getShapeWeights(grid, x, y):
    weights = [0 for _ in range(len(shapes))]
    array = [[(x,y)], [(x,y), (x + 1, y + 1)], [(x,y), (x+1, y-1)], [(x,y), (x+1, y+1), (x+2, y), (x+3, y+1)], [(x,y), (x+1, y-1), (x+2, y), (x+3, y-1)], [(x,y), (x+1, y+1), (x, y+2), (x+1, y+3)], [(x,y), (x-1, y+1), (x,y+2), (x-1, y+3)], [(x,y), (x-1, y+1), (x, y+2)], [(x,y), (x+1, y+1), (x, y+2)]]

    for i in range(len(shapes)):
        isValidShape = True
        for point in array[i]:
            x,y = point 
            if x < 0 or x >= gridSize or y < 0 or y >= gridSize:
                isValidShape = False
                break
        if isValidShape:
            for point in array[i]:
                if grid[point[0]][point[1]] != -1:
                    weights[i] += 1

    for i in range(len(shapes)):
        weights[i] = weights[i] / len(array[i])

    return weights

def getRandomShape(grid, x, y):
    return np.random.choice(len(shapes))
    # weights = getShapeWeights(grid, x, y)
    # print(weights)
    # total_weight = sum(weights)
    # if total_weight == 0:
    #     return 0
    # probabilities = [w / total_weight for w in weights]
    # return np.random.choice(len(shapes), p=probabilities)
    # weights = np.arange(1, len(shapes) + 1)
    # probabilities = weights / weights.sum()
    # return np.random.choice(len(shapes), p=probabilities)

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
    print('printing indices')
    print(indices)

    # Check if there are any empty positions available
    if not indices:
        return None  # Return None if there are no empty cells
    
    # Randomly select one of the indices
    return random.choice(indices)

# get neighbors function
def getRandomNeighbor(s):
    attempts = 0 
    max_attempts = 200
    while attempts <= max_attempts:
        i, j = getRandomEmptyPosition()
        # print('got random position')
        # move to position 
        moveToCell(i, j)
        print('random empty position')
        print(str((i, j)))

        # get current grid and bursh position
        shapePos, _, _, grid, _, _ = execute('export')
        # print('got current state')

        randomColorIndex = getAvailableColor(grid, shapePos[0], shapePos[1])
        randomShapeIndex = 0
        # if attempts == max_attempts:
        #     randomShapeIndex = 0
        #     print('max number of attempts reached')
        # else:
        #     randomShapeIndex = getRandomShape(grid, shapePos[0], shapePos[1])
        # print('random color: ' + str(randomColorIndex))
        # print('random shape index: ' + str(randomShapeIndex))
        # switch to random shape in env
        switchToShape(randomShapeIndex)

        # switching to random color in env
        switchToColor(randomColorIndex)

        if canPlace(grid, shapes[randomShapeIndex], shapePos) or attempts == max_attempts:
            print('can place neighbor')
            execute('place')
            return execute('export')
        else:
            print('cannot place this neighbor')
        attempts += 1
    return execute('export')
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

def hillClimbingWithRestarts(s, max_restarts=10, max_no_improve=100):
    current = s
    restart_count = 0
    no_improvement = 0

    while restart_count < max_restarts:
        # Get the current state and calculate its value
        _, _, _, grid, placedShapes, currentDone = current
        current_value = calculateObjectiveFunction(grid, placedShapes)

        if currentDone or checkGrid(grid):
            return current
        
        print("trying to get a random neighbor")
        neighbor = getRandomNeighbor(current)
        print('got a random neighbor')
        _, _, _, neighbor_grid, neighbor_placed_shapes, done = neighbor

        if done or checkGrid(neighbor_grid): 
            print('neighbor is solution')
            return neighbor
        
        neighbor_value = calculateObjectiveFunction(neighbor_grid, neighbor_placed_shapes)

        if neighbor_value < current_value:
            print('neighbor improves objective function: accept as solution')
            current = neighbor
            no_improvement = 0  # Reset no improvement counter
        else:
            print('neighbor worsens objective function: backtrack')
            execute('undo')
            current = execute('export')
            no_improvement += 1

        # If no improvement for too long, restart the algorithm
        if no_improvement >= max_no_improve:
            print(f'Restarting after {no_improvement} iterations without improvement.')
            current = randomRestart()  # Restart with a new random state
            no_improvement = 0
            restart_count += 1

    return current

def randomRestart():
    # Restart with a random valid initial state
    print("Random restart triggered!")
    # Execute the initial state or reset parts of the grid as needed
    shapePos, _, _, grid, neighbor_placed_shapes, done = execute('export')
    oneXoneShape = 0 
    randomColorIndex = getAvailableColor(grid, shapePos[0], shapePos[1])

    pos = getRandomEmptyPosition()
    moveToCell(pos[0], pos[1])
    switchToColor(randomColorIndex)
    switchToShape(oneXoneShape)
    
    execute('place')
    return execute('export')

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
