# How to Use setup_and_run_bot.ipynb

## ✅ Yes, you run `setup_and_run_bot.ipynb`!

This notebook is your **one-stop solution** to run the bot. Here's how:

## 📋 Step-by-Step Process

### **Step 1: Push Your Code to GitHub**

Before running the notebook, make sure your bot code is on GitHub:

1. Create a new repository on GitHub
2. Upload all your bot files:
   - `bot.py` (or `colab_bot.py`)
   - `sudoku_solver.py`
   - `image_processor.py`
   - `requirements.txt`
   - `README.md` (optional)
3. Copy your repository URL (HTTPS format, e.g., `https://github.com/yourusername/sudoku-bot.git`)

### **Step 2: Open the Notebook**

- **Google Colab**: Upload `setup_and_run_bot.ipynb` to Colab
- **Local Jupyter**: Open it in Jupyter Notebook/Lab

### **Step 3: Run the Notebook Cells**

The notebook has 6 cells. Run them **in order**:

#### **Cell 1: Configuration** 🔧
- **Edit** the `GITHUB_URL` variable with your repository URL
- **Enter** your Telegram bot token when prompted (hidden input)
- This cell stores your token and GitHub URL for later cells

#### **Cell 2: Check Environment** ✅
- Automatically detects if you're running on Colab
- Just run it - no input needed

#### **Cell 3: Clone Repository** 📥
- Clones your GitHub repository into a `repo/` folder
- If already cloned, pulls latest changes
- Shows all files in the repository

#### **Cell 4: Install Dependencies** 📦
- Installs all Python packages from `requirements.txt` (if available)
- Or installs default packages
- On Colab: automatically installs Tesseract OCR
- On local: skips Tesseract (install manually if needed)

#### **Cell 5: Run the Bot** 🤖
- Starts the Telegram bot
- The bot will run until you **stop the cell** (interrupt)
- On Colab: uses `colab_bot.py` if available, else `bot.py`
- On local: uses `bot.py`

## 🎯 Quick Summary

1. **Push code to GitHub** → Get HTTPS URL
2. **Open notebook** → Upload to Colab or open locally
3. **Edit Cell 1** → Add your GitHub URL
4. **Run all cells** → Enter token when prompted
5. **Bot runs!** → Stop cell to stop bot

## ⚠️ Important Notes

- **Token Security**: The token is stored in environment variables, not saved in the notebook
- **Colab Timeout**: Colab sessions timeout after ~90 minutes of inactivity
- **Stop Bot**: Press the stop button in the notebook to stop the bot
- **Re-run**: You can re-run cells individually if needed (but keep order)

## 🔄 Next Time

If you've already cloned the repo:
- Cell 3 will pull latest changes instead of cloning again
- You can skip to Cell 4 if dependencies are already installed
- Just run Cell 5 to start the bot

## 💡 Tips

- Keep the bot running cell active to prevent Colab timeout
- Send messages to your bot periodically to keep it active
- The notebook will remember your token and URL for the session

---

**That's it!** Just run `setup_and_run_bot.ipynb` and follow the cells. 🚀

