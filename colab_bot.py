"""
Telegram Sudoku Bot - Colab Compatible Version

This version includes workarounds for Google Colab environment:
- Uses Colab-compatible file paths
- Handles Colab's file system differently
- Includes keep-alive mechanism
"""

import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sudoku_solver import solve_sudoku
from image_processor import extract_sudoku_from_image, create_solved_image
import time

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get bot token from environment variable
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')

# Colab-specific: Create temp directory if it doesn't exist
TEMP_DIR = '/content/temp_sudoku'
os.makedirs(TEMP_DIR, exist_ok=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        '👋 Hi! I\'m a Sudoku solver bot!\n\n'
        'Send me a photo of a Sudoku puzzle and I\'ll solve it for you!\n\n'
        'Just upload a clear image of a Sudoku grid and I\'ll do the rest. 🧩\n\n'
        '⚠️ Running on Google Colab - session may timeout after inactivity'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        '📖 How to use:\n\n'
        '1. Take a clear photo of a Sudoku puzzle\n'
        '2. Send the photo to this bot\n'
        '3. Wait for me to solve it!\n'
        '4. I\'ll send back the solved puzzle with all numbers filled in\n\n'
        '💡 Tips for best results:\n'
        '• Make sure the image is clear and well-lit\n'
        '• The Sudoku grid should be clearly visible\n'
        '• Try to get the grid as straight as possible\n\n'
        '⚙️ Running on Google Colab'
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages."""
    try:
        # Send processing message
        processing_msg = await update.message.reply_text('🔍 Processing your Sudoku puzzle...')
        
        # Get the largest photo size
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # Download the photo to Colab temp directory
        photo_path = os.path.join(TEMP_DIR, f'temp_{update.message.from_user.id}_{photo.file_id}.jpg')
        await file.download_to_drive(photo_path)
        
        # Extract Sudoku grid from image
        await processing_msg.edit_text('📸 Extracting Sudoku grid from image...')
        grid = extract_sudoku_from_image(photo_path)
        
        if grid is None:
            await processing_msg.edit_text(
                '❌ Sorry, I couldn\'t extract the Sudoku puzzle from the image.\n\n'
                'Please make sure:\n'
                '• The image is clear and well-lit\n'
                '• The Sudoku grid is clearly visible\n'
                '• The grid is not too rotated or distorted'
            )
            if os.path.exists(photo_path):
                os.remove(photo_path)
            return
        
        # Solve the Sudoku
        await processing_msg.edit_text('🧮 Solving the puzzle...')
        solved_grid = solve_sudoku(grid.copy())
        
        if solved_grid is None:
            await processing_msg.edit_text(
                '❌ Sorry, I couldn\'t solve this puzzle. It might be invalid or unsolvable.'
            )
            if os.path.exists(photo_path):
                os.remove(photo_path)
            return
        
        # Create solved image
        await processing_msg.edit_text('🎨 Creating solved image...')
        solved_image_path = create_solved_image(photo_path, grid, solved_grid)
        
        if solved_image_path is None:
            await processing_msg.edit_text('❌ Error creating solved image.')
            if os.path.exists(photo_path):
                os.remove(photo_path)
            return
        
        # Send the solved image
        await processing_msg.edit_text('✅ Done! Here\'s your solved Sudoku:')
        with open(solved_image_path, 'rb') as solved_image:
            await update.message.reply_photo(
                photo=solved_image,
                caption='✅ Sudoku solved! 🎉'
            )
        
        # Clean up temporary files
        if os.path.exists(photo_path):
            os.remove(photo_path)
        if os.path.exists(solved_image_path):
            os.remove(solved_image_path)
            
    except Exception as e:
        logger.error(f"Error processing photo: {e}", exc_info=True)
        await update.message.reply_text(
            '❌ An error occurred while processing your image. Please try again with a clearer image.'
        )
        # Clean up
        photo_path = os.path.join(TEMP_DIR, f'temp_{update.message.from_user.id}_{photo.file_id}.jpg')
        if os.path.exists(photo_path):
            os.remove(photo_path)

async def keep_alive(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Keep-alive function to prevent Colab timeout."""
    # This runs periodically to keep the session active
    logger.info("Bot is running... (keep-alive)")

def main() -> None:
    """Start the bot."""
    if not BOT_TOKEN:
        print("ERROR: Please set TELEGRAM_BOT_TOKEN environment variable")
        print("You can get a token from @BotFather on Telegram")
        print("\nIn Colab, run: os.environ['TELEGRAM_BOT_TOKEN'] = 'your_token'")
        return
    
    # Create the Application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    # Add keep-alive job (runs every 5 minutes)
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(keep_alive, interval=300, first=60)
    
    # Start the bot
    print("Bot is starting...")
    print("Note: This will run until you interrupt the cell (Ctrl+C) or Colab times out")
    print("Colab sessions typically timeout after 90 minutes of inactivity")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()

