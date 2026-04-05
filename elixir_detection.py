import cv2
import numpy as np
import pygetwindow as gw
import pyautogui


def count_elixer():
    windows = gw.getWindowsWithTitle("BlueStacks")
    if not windows:
        print("BlueStacks window not found")
        return 0

    win = windows[0]
    left, top, width, height = win.left, win.top, win.width, win.height

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    h, w, _ = img.shape

    roi = img[int(h * 0.8):h, 0:w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_purple = np.array([130, 80, 80])
    upper_purple = np.array([170, 255, 255])

    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    elixir = 0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest)
        elixir = int((w_box / (w * 0.7)) * 10)
        elixir = max(0, min(10, elixir))

    return elixir


if __name__ == "__main__":
    value = count_elixer()
    print("Elixir:", value)