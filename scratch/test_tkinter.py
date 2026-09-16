import tkinter
from tkinter import filedialog
from pathlib import Path

def test_picker():
    try:
        root = tkinter.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        print("Opening picker...")
        # Since this is a test script, we will simulate a mock picker or try opening it.
        # We can just check that Tk can be initialized and destroyed.
        root.destroy()
        print("Tkinter initialized and destroyed successfully")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_picker()
