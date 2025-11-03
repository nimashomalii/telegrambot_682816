"""
Sudoku solver using backtracking algorithm
"""
import numpy as np

def is_valid(grid, row, col, num):
    """Check if placing num at (row, col) is valid."""
    # Check row
    for x in range(9):
        if grid[row][x] == num:
            return False
    
    # Check column
    for x in range(9):
        if grid[x][col] == num:
            return False
    
    # Check 3x3 box
    start_row = row - row % 3
    start_col = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + start_row][j + start_col] == num:
                return False
    
    return True

def solve_sudoku(grid):
    """
    Solve Sudoku using backtracking algorithm.
    
    Args:
        grid: 9x9 numpy array or list of lists with 0 for empty cells
    
    Returns:
        Solved 9x9 numpy array or None if unsolvable
    """
    # Convert to numpy array if needed
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=int)
    
    # Find empty cell
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                # Try numbers 1-9
                for num in range(1, 10):
                    if is_valid(grid, row, col, num):
                        grid[row][col] = num
                        
                        # Recursively solve
                        if solve_sudoku(grid) is not None:
                            return grid
                        
                        # Backtrack
                        grid[row][col] = 0
                
                return None  # No solution found
    
    return grid  # Puzzle solved

def is_valid_puzzle(grid):
    """Check if the initial puzzle is valid."""
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=int)
    
    for row in range(9):
        for col in range(9):
            if grid[row][col] != 0:
                num = grid[row][col]
                grid[row][col] = 0
                if not is_valid(grid, row, col, num):
                    grid[row][col] = num
                    return False
                grid[row][col] = num
    
    return True

