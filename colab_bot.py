"""
Telegram Sudoku Bot - Colab Compatible Version

This version includes workarounds for Google Colab environment:
- Uses Colab-compatible file paths
- Handles Colab's file system differently
- Includes keep-alive mechanism
- Works with Jupyter/Colab event loops
"""

import os
import sys
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sudoku_solver import solve_sudoku
from image_processor import extract_sudoku_from_image, create_solved_image
import time

# Fix for Jupyter/Colab event loop issues
try:
    import nest_asyncio
    nest_asyncio.apply()
    print("✅ nest_asyncio applied - Colab/Jupyter compatible")
except ImportError:
    print("⚠️  nest_asyncio not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nest_asyncio"])
    import nest_asyncio
    nest_asyncio.apply()
    print("✅ nest_asyncio installed and applied")

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get bot token from environment variable
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')

# Colab-specific: Create temp directory if it doesn't exist
# Detect if running in Colab
if 'google.colab' in sys.modules:
    TEMP_DIR = '/content/temp_sudoku'
else:
    # For local Jupyter or other environments
    TEMP_DIR = os.path.join(os.getcwd(), 'temp_sudoku')
os.makedirs(TEMP_DIR, exist_ok=True)
print(f"📁 Temporary directory: {TEMP_DIR}")

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
        
        # Validate the puzzle before solving
        from sudoku_solver import is_valid_puzzle, count_filled_cells, get_validation_info
        
        await processing_msg.edit_text('🔍 Validating puzzle...')
        
        validation_info = get_validation_info(grid)
        filled_count = validation_info['filled']
        filled_percentage = validation_info['percentage']
        
        # Check if puzzle is valid
        if not validation_info['is_valid']:
            error_msg = '❌ پازل نامعتبر است!\n\n'
            error_msg += f'📊 تعداد خانه‌های پر شده: {filled_count} از 81 ({filled_percentage:.1f}%)\n\n'
            
            if validation_info['row_errors']:
                error_msg += f'⚠️ خطا در ردیف‌های: {validation_info["row_errors"]}\n'
            if validation_info['col_errors']:
                error_msg += f'⚠️ خطا در ستون‌های: {validation_info["col_errors"]}\n'
            if validation_info['box_errors']:
                error_msg += f'⚠️ خطا در جعبه‌های: {validation_info["box_errors"]}\n'
            
            error_msg += '\n💡 مشکل احتمالی:\n'
            error_msg += '• تصویر واضح نبوده و OCR اعداد را اشتباه تشخیص داده\n'
            error_msg += '• یا بعضی اعداد تشخیص داده نشده‌اند\n'
            error_msg += '• لطفاً تصویر واضح‌تری بفرستید'
            
            await processing_msg.edit_text(error_msg)
            if os.path.exists(photo_path):
                os.remove(photo_path)
            return
        
        # Check if too many cells are empty (OCR might have failed)
        if filled_count < 17:  # Sudoku typically needs at least 17 clues
            await processing_msg.edit_text(
                '⚠️ تعداد خانه‌های خالی خیلی زیاد است!\n\n'
                f'📊 تعداد خانه‌های پر شده: {filled_count} از 81\n\n'
                '💡 ممکن است OCR اعداد را به درستی تشخیص نداده باشد.\n'
                '• لطفاً تصویر واضح‌تر و با نور بهتر بفرستید\n'
                '• مطمئن شوید که اعداد در تصویر واضح هستند'
            )
            if os.path.exists(photo_path):
                os.remove(photo_path)
            return
        
        # Solve the Sudoku
        await processing_msg.edit_text('🧮 Solving the puzzle...')
        solved_grid = solve_sudoku(grid.copy())
        
        if solved_grid is None:
            await processing_msg.edit_text(
                '❌ متأسفم، نتوانستم پازل را حل کنم.\n\n'
                f'📊 تعداد خانه‌های پر شده: {filled_count} از 81\n\n'
                '💡 ممکن است:\n'
                '• پازل نامعتبر یا غیرقابل حل باشد\n'
                '• یا OCR اعداد را اشتباه تشخیص داده باشد\n\n'
                '• لطفاً تصویر واضح‌تری بفرستید'
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
    
    # Add keep-alive job (runs every 5 minutes) - only if job_queue is available
    try:
        job_queue = application.job_queue
        if job_queue:
            job_queue.run_repeating(keep_alive, interval=300, first=60)
            print("✅ Keep-alive job scheduled")
    except Exception as e:
        print(f"⚠️  JobQueue not available (this is OK): {e}")
    
    # Start the bot
    print("Bot is starting...")
    print("Note: This will run until you interrupt the cell (Ctrl+C) or Colab times out")
    print("Colab sessions typically timeout after 90 minutes of inactivity")
    
    # Use run_polling which now works with nest_asyncio
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main()

