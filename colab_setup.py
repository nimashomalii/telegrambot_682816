"""
Google Colab setup script for Telegram Sudoku Bot

This script adapts the bot for Google Colab environment.
Run this in a Colab cell to install dependencies and set up the bot.
"""

# Install dependencies
print("Installing dependencies...")
!pip install -q python-telegram-bot==20.7 opencv-python==4.8.1.78 numpy==1.24.3 Pillow==10.1.0 pytesseract==0.3.10

# Install Tesseract OCR
print("Installing Tesseract OCR...")
!sudo apt-get -qq update
!sudo apt-get -qq install tesseract-ocr

# Upload files to Colab
print("\n" + "="*50)
print("IMPORTANT: You need to upload the following files to Colab:")
print("1. bot.py")
print("2. sudoku_solver.py")
print("3. image_processor.py")
print("\nUse the file upload button in Colab sidebar")
print("="*50)

# Set environment variable (you'll need to enter your token)
import os
print("\nEnter your Telegram bot token:")
token = input("Token: ").strip()
os.environ['TELEGRAM_BOT_TOKEN'] = token

print("\nSetup complete! Now you can run the bot.")
print("Note: The bot will run until the Colab session times out or you interrupt it.")

