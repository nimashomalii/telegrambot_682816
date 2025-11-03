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
    """Extract digit from a cell using improved OCR with better preprocessing."""
    # Remove borders more carefully
    h, w = cell.shape
    if h < 10 or w < 10:
        return 0
    
    # Remove borders (keep more of the cell)
    cell = cell[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
    
    if cell.size == 0:
        return 0
    
    # Count non-zero pixels
    non_zero = cv2.countNonZero(cell)
    total_pixels = cell.size
    
    # If too few pixels, cell is empty (lower threshold)
    if non_zero < total_pixels * 0.02:
        return 0
    
    # Try OCR with pytesseract (improved preprocessing)
    try:
        import pytesseract
        
        # Method 1: Direct OCR with improved preprocessing
        # Enhance contrast and brightness
        cell_enhanced = cv2.convertScaleAbs(cell, alpha=2.0, beta=30)
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cell_enhanced = cv2.morphologyEx(cell_enhanced, cv2.MORPH_CLOSE, kernel)
        cell_enhanced = cv2.morphologyEx(cell_enhanced, cv2.MORPH_OPEN, kernel)
        
        # Resize significantly larger for better OCR (minimum 200x200)
        min_size = 200
        scale = max(min_size / cell_enhanced.shape[0], min_size / cell_enhanced.shape[1])
        new_w = int(cell_enhanced.shape[1] * scale)
        new_h = int(cell_enhanced.shape[0] * scale)
        
        cell_resized = cv2.resize(cell_enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to PIL Image
        cell_pil = Image.fromarray(cell_resized)
        
        # Try multiple PSM modes for better recognition
        psm_modes = [
            '--psm 10',  # Single character
            '--psm 8',   # Single word
            '--psm 7',   # Single text line
        ]
        
        for psm in psm_modes:
            config = f'{psm} -c tessedit_char_whitelist=123456789'
            text = pytesseract.image_to_string(cell_pil, config=config)
            text = text.strip()
            
            if text:
                # Try to extract digit
                digits = [c for c in text if c.isdigit() and '1' <= c <= '9']
                if digits:
                    digit = int(digits[0])
                    if 1 <= digit <= 9:
                        return digit
        
        # Method 2: Try with inverted image (sometimes works better)
        cell_inv = cv2.bitwise_not(cell_enhanced)
        cell_inv_resized = cv2.resize(cell_inv, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        cell_inv_pil = Image.fromarray(cell_inv_resized)
        
        for psm in psm_modes:
            config = f'{psm} -c tessedit_char_whitelist=123456789'
            text = pytesseract.image_to_string(cell_inv_pil, config=config)
            text = text.strip()
            
            if text:
                digits = [c for c in text if c.isdigit() and '1' <= c <= '9']
                if digits:
                    digit = int(digits[0])
                    if 1 <= digit <= 9:
                        return digit
        
        # Method 3: Try with adaptive threshold
        cell_adaptive = cv2.adaptiveThreshold(
            cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        cell_adaptive_resized = cv2.resize(cell_adaptive, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        cell_adaptive_pil = Image.fromarray(cell_adaptive_resized)
        
        for psm in psm_modes:
            config = f'{psm} -c tessedit_char_whitelist=123456789'
            text = pytesseract.image_to_string(cell_adaptive_pil, config=config)
            text = text.strip()
            
            if text:
                digits = [c for c in text if c.isdigit() and '1' <= c <= '9']
                if digits:
                    digit = int(digits[0])
                    if 1 <= digit <= 9:
                        return digit
                        
    except Exception as e:
        # If OCR fails, try contour-based detection as fallback
        pass
    
    # Fallback: Contour-based detection (if OCR completely fails)
    try:
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            cell_area = cell.shape[0] * cell.shape[1]
            
            # If contour is significant, there's likely a digit (but we can't read it)
            if area > cell_area * 0.15:
                # Return -1 to indicate digit exists but couldn't be read
                # This helps debug which cells have numbers
                return -1
    except:
        pass
    
    # If all methods fail, return 0 (empty cell)
    return 0

def extract_sudoku_from_image(image_path):
    """
    Extract Sudoku grid from image with improved preprocessing.
    
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
    
    # Enhance image quality
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Try multiple threshold methods and use the best one
    # Method 1: Adaptive threshold
    thresh1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Method 2: Otsu's threshold
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Method 3: Simple threshold (try different values)
    _, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Divide into 9x9 grid
    h, w = gray.shape
    cell_h = h // 9
    cell_w = w // 9
    
    grid = np.zeros((9, 9), dtype=int)
    
    # Extract digits from each cell - try multiple threshold methods
    for i in range(9):
        for j in range(9):
            y1 = max(0, i * cell_h)
            y2 = min(h, (i + 1) * cell_h)
            x1 = max(0, j * cell_w)
            x2 = min(w, (j + 1) * cell_w)
            
            # Try with adaptive threshold first
            cell1 = thresh1[y1:y2, x1:x2]
            digit = extract_digit(cell1)
            
            # If failed, try with Otsu threshold
            if digit == 0 or digit == -1:
                cell2 = thresh2[y1:y2, x1:x2]
                digit2 = extract_digit(cell2)
                if digit2 > 0:
                    digit = digit2
            
            # If still failed, try with simple threshold
            if digit == 0 or digit == -1:
                cell3 = thresh3[y1:y2, x1:x2]
                digit3 = extract_digit(cell3)
                if digit3 > 0:
                    digit = digit3
            
            # If digit is -1 (detected but couldn't read), set to 0
            if digit == -1:
                digit = 0
            
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

