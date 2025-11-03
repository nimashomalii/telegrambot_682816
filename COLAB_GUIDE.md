# Running Telegram Sudoku Bot on Google Colab

Yes, you can run this bot on Google Colab! However, there are some important considerations:

## ⚠️ Limitations

1. **Session Timeout**: Colab sessions timeout after ~90 minutes of inactivity
2. **Not Persistent**: You need to restart the bot each time you open Colab
3. **Resource Limits**: Colab has CPU/memory limits (though sufficient for this bot)

## 📋 Setup Steps

### Step 1: Open Google Colab
Go to [Google Colab](https://colab.research.google.com/)

### Step 2: Install Dependencies

Create a new cell and run:

```python
# Install Python packages
!pip install -q python-telegram-bot==20.7 opencv-python==4.8.1.78 numpy==1.24.3 Pillow==10.1.0 pytesseract==0.3.10

# Install Tesseract OCR
!sudo apt-get -qq update
!sudo apt-get -qq install tesseract-ocr
```

### Step 3: Upload Files

You need to upload these files to Colab:
1. `bot.py` (or use `colab_bot.py` - Colab-optimized version)
2. `sudoku_solver.py`
3. `image_processor.py`

**Option A: Manual Upload**
- Click the folder icon in Colab sidebar
- Click "Upload" and select the files

**Option B: Use files from GitHub/Drive**
```python
# If files are in a GitHub repo
!git clone https://github.com/yourusername/sudoku-bot.git
%cd sudoku-bot

# Or upload directly from URL
!wget https://raw.githubusercontent.com/yourusername/sudoku-bot/main/bot.py
!wget https://raw.githubusercontent.com/yourusername/sudoku-bot/main/sudoku_solver.py
!wget https://raw.githubusercontent.com/yourusername/sudoku-bot/main/image_processor.py
```

### Step 4: Set Your Bot Token

```python
import os

# Get your token from @BotFather on Telegram
os.environ['TELEGRAM_BOT_TOKEN'] = 'your_bot_token_here'
```

### Step 5: Run the Bot

```python
# Use the Colab-optimized version
from colab_bot import main
main()

# OR use the regular version
from bot import main
main()
```

## 🔄 Keep Session Alive

To prevent Colab from timing out, you can:

1. **Keep the cell running** - Don't interrupt it
2. **Use the keep-alive** - The `colab_bot.py` includes a keep-alive mechanism
3. **Interact periodically** - Send messages to your bot every so often

## 💡 Tips

- **Use `colab_bot.py`**: It's optimized for Colab with better error handling
- **Monitor the output**: Check for any errors in the Colab cell
- **Test first**: Send a test image to make sure everything works
- **Save your work**: Download any important files before the session ends

## 🚀 Alternative: Use Replit or Railway

For a more persistent solution, consider:
- **Replit**: Free tier with persistent processes
- **Railway**: Free tier for hobby projects
- **Heroku**: Alternative cloud platform
- **VPS**: DigitalOcean, AWS, etc.

## 📝 Example Colab Notebook

Here's a complete example you can paste into Colab:

```python
# Install dependencies
!pip install -q python-telegram-bot==20.7 opencv-python==4.8.1.78 numpy==1.24.3 Pillow==10.1.0 pytesseract==0.3.10
!sudo apt-get -qq update
!sudo apt-get -qq install tesseract-ocr

# Set token (replace with your token)
import os
os.environ['TELEGRAM_BOT_TOKEN'] = 'YOUR_TOKEN_HERE'

# Upload files manually or use !wget/!git clone

# Run bot
from colab_bot import main
main()
```

## ⚡ Quick Start Script

You can also use the `colab_setup.py` file as a reference for the setup process.

