import pyautogui
import time
import os
import platform
from pynput import keyboard
import pygetwindow as gw

def continuous_screenshot(save_folder="TRAIN", interval=0.15):
    # Create folder
    os.makedirs(save_folder, exist_ok=True)

    # Detect BlueStacks window
    windows = gw.getWindowsWithTitle("BlueStacks")
    if not windows:
        print("BlueStacks not found!")
        return

    win = windows[0]

    region = (
        win.left,
        win.top,
        win.width,
        win.height
    )

    print(f"Capturing region: {region}")

    stop_flag = {"stop": False}

    # Key listener
    def on_press(key):
        try:
            if key.char == 'q':
                print("\nStopping capture...")
                stop_flag["stop"] = True
                return False
        except:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Started capturing... Press 'q' to stop.")

    count = 0

    while not stop_flag["stop"]:
        screenshot = pyautogui.screenshot(region=region)

        filename = os.path.join(save_folder, f"frame_{count:05d}.png")
        screenshot.save(filename)

        count += 1
        time.sleep(interval)

    listener.join()
    print(f"Saved {count} screenshots.")


if __name__ == '__main__':
    continuous_screenshot()