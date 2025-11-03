"""
Image processing module for Sudoku extraction and image generation
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def preprocess_image(image_path):
    """Preprocess image for better Sudoku detection."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return thresh, img

def find_sudoku_grid(image_path):
    """Find the Sudoku grid in the image."""
    thresh, original = preprocess_image(image_path)
    if thresh is None:
        return None, None
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (likely the grid)
    if not contours:
        return None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we found a quadrilateral
    if len(approx) == 4:
        return approx, original
    
    # If not, try to find a rectangular contour
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return approx, original
    
    return None, original

def order_points(pts):
    """Order points in top-left, top-right, bottom-right, bottom-left order."""
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def perspective_transform(image, pts):
    """Apply perspective transform to get a top-down view."""
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect
    
    # Compute width and height
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def extract_digit(cell):
    """Extract digit from a cell using simple threshold and OCR-like approach."""
    # Remove borders
    h, w = cell.shape
    cell = cell[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    
    if cell.size == 0:
        return 0
    
    # Count non-zero pixels
    non_zero = cv2.countNonZero(cell)
    total_pixels = cell.size
    
    # If too few pixels, cell is empty
    if non_zero < total_pixels * 0.05:
        return 0
    
    # Simple digit recognition based on pixel patterns
    # This is a simplified approach - for better accuracy, use OCR library
    
    # For now, use a basic approach: detect if there are enough pixels to be a digit
    # This is a placeholder - in production, use pytesseract or a trained model
    
    # Try to extract using contour detection
    contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    
    # Find largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    cell_area = cell.shape[0] * cell.shape[1]
    
    # If the contour is too small, likely empty
    if area < cell_area * 0.1:
        return 0
    
    # For now, return a placeholder - user will need to implement OCR
    # or use a digit recognition model
    # This is where you'd integrate pytesseract or a CNN model
    
    # Simple heuristic: try to read with pytesseract if available
    try:
        import pytesseract
        # Convert to PIL Image for tesseract
        cell_pil = Image.fromarray(cell)
        # Resize for better recognition
        cell_pil = cell_pil.resize((cell_pil.width * 4, cell_pil.height * 4), Image.LANCZOS)
        text = pytesseract.image_to_string(cell_pil, config='--psm 10 -c tessedit_char_whitelist=123456789')
        text = text.strip()
        if text and text.isdigit() and 1 <= int(text) <= 9:
            return int(text)
    except:
        pass
    
    # If OCR fails, return 0 (will need manual input or better OCR)
    return 0

def extract_sudoku_from_image(image_path):
    """
    Extract Sudoku grid from image.
    
    Returns:
        9x9 numpy array with 0 for empty cells, or None if extraction fails
    """
    # Find grid
    grid_pts, original = find_sudoku_grid(image_path)
    if grid_pts is None:
        return None
    
    # Apply perspective transform
    warped = perspective_transform(original, grid_pts)
    
    # Convert to grayscale
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Divide into 9x9 grid
    h, w = thresh.shape
    cell_h = h // 9
    cell_w = w // 9
    
    grid = np.zeros((9, 9), dtype=int)
    
    # Extract digits from each cell
    for i in range(9):
        for j in range(9):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w
            
            cell = thresh[y1:y2, x1:x2]
            digit = extract_digit(cell)
            grid[i][j] = digit
    
    return grid

def create_solved_image(original_image_path, initial_grid, solved_grid):
    """
    Create an image showing the solved Sudoku.
    
    Args:
        original_image_path: Path to original image
        initial_grid: 9x9 array with initial puzzle
        solved_grid: 9x9 array with solved puzzle
    
    Returns:
        Path to the solved image
    """
    # Load original image
    img = cv2.imread(original_image_path)
    if img is None:
        return None
    
    # Find grid for perspective transform
    grid_pts, original = find_sudoku_grid(original_image_path)
    if grid_pts is None:
        return None
    
    # Get warped view
    warped = perspective_transform(original, grid_pts)
    h, w = warped.shape[:2]
    
    # Create a new image with solution (start with original warped)
    solved_img = warped.copy()
    
    # Calculate cell dimensions
    cell_h = h // 9
    cell_w = w // 9
    
    # Draw existing numbers in black, new numbers in blue
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Scale font based on cell size
    font_scale = min(cell_w, cell_h) / 40.0
    thickness = max(1, int(font_scale * 2))
    
    for i in range(9):
        for j in range(9):
            if solved_grid[i][j] != 0:
                # Check if this was in the original puzzle
                was_empty = initial_grid[i][j] == 0
                
                # Calculate position
                x = j * cell_w + cell_w // 2
                y = i * cell_h + cell_h // 2
                
                # Color: blue for solved numbers, keep original numbers
                if was_empty:
                    # Draw filled background for new numbers
                    cv2.rectangle(solved_img, 
                                (j * cell_w + 5, i * cell_h + 5),
                                ((j + 1) * cell_w - 5, (i + 1) * cell_h - 5),
                                (200, 220, 255), -1)  # Light blue background
                    color = (255, 100, 0)  # Orange/blue for new numbers
                else:
                    color = (0, 0, 0)  # Black for original numbers
                
                # Draw number
                text = str(solved_grid[i][j])
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2
                
                cv2.putText(solved_img, text, (text_x, text_y), 
                           font, font_scale, color, thickness)
    
    # Transform back to original perspective
    rect = order_points(grid_pts.reshape(4, 2))
    (tl, tr, br, bl) = rect
    
    # Compute width and height of destination
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Get inverse transform matrix
    M_inv = cv2.getPerspectiveTransform(dst, rect)
    
    # Transform solved image back to original perspective
    h_orig, w_orig = original.shape[:2]
    solved_warped = cv2.warpPerspective(solved_img, M_inv, (w_orig, h_orig))
    
    # Create mask for the grid area
    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    cv2.fillPoly(mask, [grid_pts.reshape(-1, 1, 2).astype(int)], 255)
    
    # Blend the solved grid back onto the original image
    result = original.copy()
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = (result * (1 - mask_3d) + solved_warped * mask_3d).astype(np.uint8)
    
    # Save the solved image
    output_path = original_image_path.replace('.jpg', '_solved.jpg')
    if not output_path.endswith('.jpg'):
        output_path += '.jpg'
    cv2.imwrite(output_path, result)
    
    return output_path

