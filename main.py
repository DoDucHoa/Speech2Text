# main.py
import tkinter as tk
from gui import WhisperGUI

def main():
    root = tk.Tk()
    app = WhisperGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
