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
        grid = np.array(grid, dtype=int).copy()
    else:
        grid = grid.copy()
    
    # First check if puzzle is valid
    if not is_valid_puzzle(grid):
        return None
    
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

def count_filled_cells(grid):
    """Count how many cells are filled (non-zero)."""
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=int)
    return np.count_nonzero(grid)

def get_validation_info(grid):
    """Get information about why a puzzle might be invalid."""
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=int)
    
    filled = count_filled_cells(grid)
    total = 81
    
    # Check for duplicate numbers in rows
    row_errors = []
    for i in range(9):
        row = grid[i]
        non_zero = row[row != 0]
        if len(non_zero) != len(np.unique(non_zero)):
            row_errors.append(i + 1)
    
    # Check for duplicate numbers in columns
    col_errors = []
    for j in range(9):
        col = grid[:, j]
        non_zero = col[col != 0]
        if len(non_zero) != len(np.unique(non_zero)):
            col_errors.append(j + 1)
    
    # Check for duplicate numbers in boxes
    box_errors = []
    for box_row in range(3):
        for box_col in range(3):
            start_r = box_row * 3
            start_c = box_col * 3
            box = grid[start_r:start_r+3, start_c:start_c+3]
            non_zero = box[box != 0]
            if len(non_zero) != len(np.unique(non_zero)):
                box_errors.append((box_row + 1, box_col + 1))
    
    return {
        'filled': filled,
        'total': total,
        'percentage': (filled / total) * 100,
        'row_errors': row_errors,
        'col_errors': col_errors,
        'box_errors': box_errors,
        'is_valid': len(row_errors) == 0 and len(col_errors) == 0 and len(box_errors) == 0
    }

