# Sudoku Solver Telegram Bot

A Telegram bot that can receive a photo of a Sudoku puzzle, solve it, and send back the solved puzzle with all numbers filled in.

## Features

- 📸 Receives Sudoku puzzle images via Telegram
- 🔍 Extracts the Sudoku grid from images using computer vision
- 🧮 Solves the puzzle using backtracking algorithm
- 🎨 Generates and sends back the solved puzzle image
- ✨ Highlights newly solved numbers in blue

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for digit recognition)

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH after installation

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 3. Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Follow the instructions to create your bot
4. Copy the bot token you receive

### 4. Set Environment Variable

**Windows (PowerShell):**
```powershell
$env:TELEGRAM_BOT_TOKEN="your_bot_token_here"
```

**Windows (Command Prompt):**
```cmd
set TELEGRAM_BOT_TOKEN=your_bot_token_here
```

**Linux/macOS:**
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
```

Or create a `.env` file (not recommended for production):
```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

### 5. Run the Bot

```bash
python bot.py
```

## Usage

1. Start a chat with your bot on Telegram
2. Send `/start` to begin
3. Send a clear photo of a Sudoku puzzle
4. Wait for the bot to process and solve it
5. Receive the solved puzzle image!

## How It Works

1. **Image Processing**: Uses OpenCV to detect and extract the Sudoku grid from the image
2. **Digit Recognition**: Uses Tesseract OCR to recognize numbers in each cell
3. **Solving**: Implements a backtracking algorithm to solve the puzzle
4. **Image Generation**: Creates a new image with the solved puzzle, highlighting new numbers in blue

## Tips for Best Results

- Take clear, well-lit photos
- Ensure the Sudoku grid is clearly visible
- Try to keep the grid as straight as possible
- Avoid shadows and reflections
- Make sure numbers are clearly visible

## Troubleshooting

**Bot not responding:**
- Check that `TELEGRAM_BOT_TOKEN` is set correctly
- Verify the bot token is valid

**Can't extract puzzle:**
- Try a clearer image
- Ensure the grid is well-defined and visible
- Check that the image shows the full Sudoku grid

**OCR not working:**
- Make sure Tesseract is installed and in PATH
- Try images with clearer, larger numbers

## License

MIT License - feel free to use and modify!

