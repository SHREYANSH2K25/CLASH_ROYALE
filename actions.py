import pyautogui
import os
from datetime import datetime
import time
import platform
import pygetwindow as gw
import cv2
import numpy as np

import pyscreeze
pyscreeze.USE_IMAGE_NOT_FOUND_EXCEPTION = False

# SPELL_CARDS = {
#     "Fireball", "Zap", "Arrows", "Tornado",
#     "Rocket", "Lightning", "Freeze"
# }


class Actions:
    def __init__(self, TOP_LEFT_X=None, TOP_LEFT_Y=None, BOTTOM_RIGHT_X=None, BOTTOM_RIGHT_Y=None):
        self.os_type = platform.system()
        self.script_dir = os.path.dirname(__file__)
        self.images_folder = os.path.join(self.script_dir, 'images')
        # stores images like winner, ok, battle....
        self.check_images = os.path.join(self.script_dir,'check_images')
        # Auto-detect BlueStacks window if coordinates not provided
        if TOP_LEFT_X is None:
            windows = gw.getWindowsWithTitle("BlueStacks")
            if not windows:
                raise RuntimeError("BlueStacks window not found. Please pass coordinates manually.")
            win = windows[0]
            TOP_LEFT_X    = win.left
            TOP_LEFT_Y    = win.top
            BOTTOM_RIGHT_X = win.left + win.width
            BOTTOM_RIGHT_Y = win.top + win.height
            print(f"Auto-detected BlueStacks: ({TOP_LEFT_X}, {TOP_LEFT_Y}) -> ({BOTTOM_RIGHT_X}, {BOTTOM_RIGHT_Y})")
            print(win.width," ",win.height)

        self.TOP_LEFT_X = TOP_LEFT_X
        self.TOP_LEFT_Y = TOP_LEFT_Y
        self.BOTTOM_RIGHT_X = BOTTOM_RIGHT_X
        self.BOTTOM_RIGHT_Y = BOTTOM_RIGHT_Y
        self.FIELD_AREA = (self.TOP_LEFT_X, self.TOP_LEFT_Y, self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y)
        
        self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
        self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y
        
        # Add card bar coordinates for Windows
        self.WIDTH  = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
        self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y

        self.CARD_BAR_X = int(self.TOP_LEFT_X + 0.24* self.WIDTH)
        self.CARD_BAR_Y = int(self.TOP_LEFT_Y + 0.82 * self.HEIGHT)

        self.CARD_BAR_WIDTH  = int(0.72 * self.WIDTH)
        self.CARD_BAR_HEIGHT = int(0.13 * self.HEIGHT)
        # print(self.CARD_BAR_X, self.CARD_BAR_Y, self.CARD_BAR_WIDTH, self.CARD_BAR_HEIGHT)
        # Card position to key mapping - maps 4 card slots to keyboard keys
        self.card_keys = {
            0: '1',  # Changed from 1 to 0
            1: '2',  # Changed from 2 to 1
            2: '3',  # Changed from 3 to 2
            3: '4'   # Changed from 4 to 3
        }
        
        # Card name to position mapping (will be updated during detection)
        self.current_card_positions = {}
        
    # saves card in save_path(if given) and returns numpy array
    # Grabs the entire BlueStacks window
    def capture_area(self, save_path=None):
        try:
            screenshot = pyautogui.screenshot(region=(self.TOP_LEFT_X, self.TOP_LEFT_Y, self.WIDTH, self.HEIGHT))
        except OSError as e:
            print(f"[capture_area] Screen grab failed: {e}. Is BlueStacks minimized?")
            return None
        if save_path:
            screenshot.save(save_path)
        return   np.array(screenshot)[:, :, ::-1]

    # Specifically crops out the bottom section where your cards are displayed
    def capture_card_area(self, save_path):
        screenshot = pyautogui.screenshot(region=(
            self.CARD_BAR_X, 
            self.CARD_BAR_Y, 
            self.CARD_BAR_WIDTH, 
            self.CARD_BAR_HEIGHT
        ))
        screenshot.save(save_path)
        return screenshot
    
    # Divides card bar into four equal pieces to look at each card slot individually
    # takes same big picture but then uses code to slice it into four smaller equal pieces.
    def capture_individual_cards(self):
        screenshot = pyautogui.screenshot(region=(
            self.CARD_BAR_X, 
            self.CARD_BAR_Y, 
            self.CARD_BAR_WIDTH, 
            self.CARD_BAR_HEIGHT
        ))
        
        # slicing
        card_width = self.CARD_BAR_WIDTH // 4
        cards = []
        for i in range(4):
            left = i * card_width
            card_img = screenshot.crop((left, 0, left + card_width, self.CARD_BAR_HEIGHT))
            save_path = os.path.join(self.script_dir, 'screenshots', f"card_{i+1}.png")
            card_img.save(save_path)
            cards.append(save_path)
        
        return cards

    def count_elixir(self):
        windows = gw.getWindowsWithTitle("BlueStacks")
        if not windows:
            print("BlueStacks window not found")
            return 0

        win = windows[0]
        left, top, width, height = win.left, win.top, win.width, win.height

        try:
            # takes snapshot of only the area covered by the BlueStack window
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
        except OSError as e:
            print(f"[count_elixir] Screen grab failed: {e}")
            return 0
        
        # Pygame takes screenshot in RGB format, but OpenCV works in BGR
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        h, w, _ = img.shape

        # elixir always at bottom of the screen. Crop only to bottom 20% 
        roi = img[int(h * 0.8):h, 0:w]

        # converts the crop to hsv [hue, saturation, Value]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # mark ranges of purple so as to adjust as per brightness of screen
        lower_purple = np.array([130, 80, 80])
        upper_purple = np.array([170, 255, 255])

        # convert every pixel within range to binary image with purple area white(255) and rest all black(0)
        mask = cv2.inRange(hsv, lower_purple, upper_purple)

        # used to detect shapes of white pixel in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        elixir = 0

        # it draws an invisible box around that purple bar and measures its width
        # width of purple is approx 70% of screen width then count 10 elexir and continue in this ratio
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest)
            elixir = int((w_box / (w * 0.7)) * 10)
            elixir = max(0, min(10, elixir))

        return elixir


# update card position based on detection which is a dictionary with card class and : position
    # expects something like this
    # dic1 = [
    # {'class': 'arpit', 'x': 3},
    # {'class': 'narpt', 'x': 2},
    # {'class': 'shr', 'x': 0},
    # {'class': 'her', 'x': 1}
    # ]
    
    # The AI doesn't just look at the screen; 
    # it maintains a "map" of which card is in which slot
    def update_card_positions(self, detections):
        # It takes a list of detections (which unit was found at which X-coordinate) 
        # then sorts them from left to right.
        sorted_cards = sorted(detections, key=lambda x: x['x'])
         
        # assign each unit class(like knight or archer) to an index from 0 to 3
        self.current_card_positions = {
            card['class']: idx  # Removed +1 
            for idx, card in enumerate(sorted_cards)
    }
    
    # plays card_index on x,y coordinate
    def card_play(self, x, y, card_index):
        print(f"Playing card {card_index} at position ({x}, {y})")
        if card_index in self.card_keys:
            
            key = self.card_keys[card_index]
            print(f"Pressing key: {key}")
            
            #check if clicks in right area 
            pyautogui.moveTo(1611, 860, duration=0.2)
            pyautogui.click()
            pyautogui.moveTo(1611, 460, duration=0.2)
            
            # select card key
            pyautogui.press(key)
            time.sleep(0.2)
            print(f"Moving mouse to: ({x}, {y})")
            # place on {x,y}
            pyautogui.moveTo(x, y, duration=0.2)
            print("Clicking")
            pyautogui.click()
        else:
            print(f"Invalid card index: {card_index}")

    # def click_battle_start(self):
    #     button_image = os.path.join(self.images_folder, "battlestartbutton.png")
    #     confidences = [0.8, 0.7, 0.6, 0.5]  # Try multiple confidence levels

    #     # Define the region (left, top, width, height) for the correct battle button
    #     battle_button_region = (1486, 755, 1730-1486, 900-755)

    #     while True:
    #         for confidence in confidences:
    #             print(f"Looking for battle start button (confidence: {confidence})")
    #             try:
    #                 location = pyautogui.locateOnScreen(
    #                     button_image,
    #                     confidence=confidence,
    #                     region=battle_button_region  # Only search in this region
    #                 )
    #                 if location:
    #                     x, y = pyautogui.center(location)
    #                     print(f"Found battle start button at ({x}, {y})")
    #                     pyautogui.moveTo(x, y, duration=0.2)
    #                     pyautogui.click()
    #                     return True
    #             except:
    #                 pass

    #         print("Button not found, clicking to clear screens...")
    #         pyautogui.moveTo(1705, 331, duration=0.2)
    #         pyautogui.click()
    #         time.sleep(1)




    # def click_battle_start(self):
    #     button_image = os.path.join(self.check_images, "battle.png")
    #     confidences = [0.8, 0.7, 0.6, 0.5]

    #     region = (
    #         # self.TOP_LEFT_X + int(0.30 * self.WIDTH),
    #         # self.TOP_LEFT_Y + int(0.70 * self.HEIGHT),
    #         # int(0.4 * self.WIDTH),
    #         # int(0.2 * self.HEIGHT)
    #         1510,
    #         730,
    #         240,
    #         150

    #     )
    #     # img = pyautogui.screenshot(region=region)
    #     # img = np.array(img)

    #     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # # these two if needed
    #     # res = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    #     # res = res.astype(np.uint8)

    #     # cv2.imshow("res", res)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     # print("region",self.TOP_LEFT_X + int(0.31 * self.WIDTH),self.TOP_LEFT_Y + int(0.73 * self.HEIGHT))

    #     start_time = time.time()    
    #     # 10 sec loop for battle image finder      
    #     while time.time() - start_time < 10:
    #         for confidence in confidences:
    #             try:
    #                 location = pyautogui.locateOnScreen(
    #                     button_image,
    #                     confidence=confidence,
    #                     region=region
    #                 )
    #                 if location:
    #                     x, y = pyautogui.center(location)
    #                     pyautogui.click(x, y)
    #                     return True
    #             except Exception as e:
    #                 print(e)
    #         # if we cannot see battle screen then something may be blocking it so , it is to click at somewhere in screen to see battle 
    #         # pyautogui.click(self.TOP_LEFT_X + 450, self.TOP_LEFT_Y + 450)
    #         # print("click region",self.TOP_LEFT_X + 200, self.TOP_LEFT_Y + 200)
    #         time.sleep(1)

    #     return False

    # def detect_game_end(self):
    #     winner_img = os.path.join(self.images_folder, "Winner.png")
    #     if not os.path.exists(winner_img):
    #         # Template missing — can't detect; skip silently
    #         return None
    #     try:
    #         confidences = [0.8, 0.7, 0.6]
    #         winner_region = (1510, 121, 1678 - 1510, 574 - 121)

    #         for confidence in confidences:
    #             winner_location = None
    #             try:
    #                 winner_location = pyautogui.locateOnScreen(
    #                     winner_img, confidence=confidence, grayscale=True, region=winner_region
    #                 )
    #             except OSError:
    #                 return None  # Screen grab failed; bail out
    #             except Exception:
    #                 pass

    #             if winner_location:
    #                 _, y = pyautogui.center(winner_location)
    #                 result = "victory" if y > 402 else "defeat"
    #                 print(f"Game end detected: {result} (y={y}, conf={confidence})")
    #                 time.sleep(3)
    #                 play_again_x, play_again_y = 1522, 913
    #                 print(f"Clicking Play Again at ({play_again_x}, {play_again_y})")
    #                 pyautogui.moveTo(play_again_x, play_again_y, duration=0.2)
    #                 pyautogui.click()
    #                 return result
    #     except Exception as e:
    #         print(f"[detect_game_end] Unexpected error: {e}")
    #     return None

    # def detect_game_end(self):
    #     winner_img = os.path.join(self.check_images, "Winner.png")

    #     if not os.path.exists(winner_img):
    #         return None

    #     # 🔹 Dynamic region (middle-top area where "Winner!" appears)
    #     winner_region = (
    #         1400,
    #         110,
    #         400,
    #         550
    #     )
    #     # scn = pyautogui.screenshot(region=(
    #     #     1400,
    #     #     110,
    #     #     400,
    #     #     550
    #     #  )) 
    #     # scn = np.array(scn) 
    #     # ck = cv2.imshow("fr",scn) 
    #     # cv2.waitKey(0) 
    #     # cv2.destroyAllWindows()
    #     # winner_img = cv2.imread(w_img) 
    #     # winner_img = cv2.cvtColor(winner_img,cv2.COLOR_BGR2GRAY)

    #     # 🔹 Dynamic threshold for victory/defeat split
    #     threshold_y = 385
    #     # print("tr",threshold_y)
    #     confidences = [0.8, 0.7, 0.6]

    #     try:
    #         for confidence in confidences:
    #             try:
    #                 location = pyautogui.locateOnScreen(
    #                     winner_img,
    #                     confidence=confidence,
    #                     grayscale=True,
    #                     region=winner_region
    #                 )
    #             except OSError:
    #                 return None
    #             except Exception as e:
    #                 print(f"[detect_game_end] locate error: {e}")
    #                 continue

    #             if location:
    #                 x, y = pyautogui.center(location)

    #                 # 🔹 Decide result dynamically
    #                 result = "victory" if y > threshold_y else "defeat"

    #                 print(f"Game end detected: {result} (y={y}, conf={confidence})")

    #                 time.sleep(2)  # wait for UI to settle

    #                 # click_x = self.TOP_LEFT_X + int(0.50 * self.WIDTH) - 30
    #                 # click_y = self.TOP_LEFT_Y + int(0.85 * self.HEIGHT)+ 60

    #                 # print(f"Clicking at ({click_x}, {click_y})")

    #                 print("CLicking of 2 for match_final(OK)")
    #                 pyautogui.moveTo(1611, 831, duration=0.2)
    #                 pyautogui.click()
    #                 pyautogui.press('2')
    #                 # pyautogui.moveTo(click_x, click_y, duration=0.2)
    #                 # pyautogui.click()
    #                 print(result)
    #                 return result

    #     except Exception as e:
    #         print(f"[detect_game_end] Unexpected error: {e}")

    #     return None

    
    # BATTLE BUTTON template matching
    def click_battle_start(self):
        button_image = os.path.join(self.check_images, "battle.png")
        confidences = [0.8, 0.7, 0.6, 0.5]

        region = (
            1510,
            730,
            240,
            150
        )
        # scn = pyautogui.screenshot(region=region)
        # scn = np.array(scn) 
        # scn = cv2.cvtColor(scn,cv2.COLOR_BGR2RGB)
        # ck = cv2.imshow("fr",scn) 
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()

        start_time = time.time()    
        clk = False
        # 10 sec loop for battle image finder      
        while time.time() - start_time < 10:
            for confidence in confidences:
                # pass target, confidencse, color blind mode(for speedy searching), search area
                location = pyautogui.locateOnScreen(
                    button_image,
                    confidence=confidence,
                    grayscale=True, # ignores color and only looks at the shapes of the letters/button
                    region=region
                )

                if location and not clk:
                    print(location)
                    #clicks at the center of the button if found
                    x, y = pyautogui.center(location)
                    print(x,y)
                    # pyautogui.moveTo(x, y, duration=0.2)
                    pyautogui.click(x, y)
                    clk = True
                    return True

            time.sleep(1)

        return False



    def detect_game_end(self):
        winner_img = os.path.join(self.check_images, "Winner.png")

        if not os.path.exists(winner_img):
            return None

        # 🔹 Dynamic region (middle-top area where "Winner!" appears)
        winner_region = (
            1400,
            110,
            400,
            550
        )

        # 🔹 Dynamic threshold for victory/defeat split
        # horizontal "dividing line" used to tell the difference between a Victory and a Defeat.
        threshold_y = 385

        confidences = [0.8, 0.7, 0.6]

        for confidence in confidences:
            location = pyautogui.locateOnScreen(
                winner_img,
                confidence=confidence,
                grayscale=True,
                region=winner_region
            )

            if location:
                x, y = pyautogui.center(location)

                # 🔹 Decide result dynamically
                result = "victory" if y > threshold_y else "defeat"

                print(f"Game end detected: {result} (y={y}, conf={confidence})")

                time.sleep(2)  # wait for UI to settle

                print("CLicking of 2 for match_final(OK)")
                
                # targetting ok button
                pyautogui.moveTo(1611, 831, duration=0.2)
                pyautogui.click()
                # Key '2' usually opens the Main Battle Screen.
                pyautogui.press('2')

                print(result)
                return result

        return None






    # def detect_match_over(self):
    #     matchover_img = os.path.join(self.check_images, "matchover.png")
    #     if not os.path.exists(matchover_img):
    #         return False
    #     # img = cv2.imread(matchover_img)
    #     # print(img is None)
    #     # # 🔹 Dynamic region (upper-middle area where "match over" appears)
    #     region = (1400, 300, 420, 200)

    #     # img = pyautogui.screenshot(region=region)
    #     # img = np.array(img)

    #     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # # these two if needed
    #     # res = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    #     # res = res.astype(np.uint8)

    #     confidences = [0.8, 0.6, 0.4]

    #     try:
    #         for confidence in confidences:
    #             try:
    #                 location = pyautogui.locateOnScreen(
    #                     matchover_img,
    #                     confidence=confidence,
    #                     grayscale=True,
    #                     region=region
    #                 )
    #             except OSError:
    #                 return False
    #             except Exception as e:
    #                 print(f"[detect_match_over]1 locate error: {e}")
    #                 continue

    #             if location:
    #                 x, y = pyautogui.center(location)
    #                 print(f"Match over detected at ({x}, {y}) with conf={confidence}")
    #                 return True

    #     except Exception as e:
    #         print(f"[detect_match_over]2 Unexpected error: {e}")

    #     return False
    def detect_match_over(self):
        matchover_img = os.path.join(self.check_images, "matchover.png")

        if not os.path.exists(matchover_img):
            return False

        region = (1400, 300, 420, 200)
        confidences = [0.8, 0.6, 0.4]

        for confidence in confidences:
            try:
                location = pyautogui.locateOnScreen(
                    matchover_img,
                    confidence=confidence,
                    grayscale=True,
                    region=region
                )
            except pyautogui.ImageNotFoundException:
                continue  # 🔹 THIS is the key fix

            if location:
                x, y = pyautogui.center(location)
                print(f"Match over detected at ({x}, {y}) with conf={confidence}")
                return True

        return False




if __name__ == '__main__':

    windows = gw.getWindowsWithTitle("BlueStack")

    if windows:
        win = windows[0]
        
        print("Top-left:", (win.left, win.top))
        print("Width, Height:", (win.width, win.height))
    else:
        print("BlueStacks not found")
    actions = Actions(win.left, win.top, win.left+win.width, win.top+win.height)
    actions.capture_area(r"C:\Users\shour\RL\CLASH_ROYALE\images\scr3.jpg")
    # actions.capture_card_area(r"C:\Users\shour\RL\CLASH_ROYALE\images\card_bar.jpg")
    # print(actions.capture_individual_cards())
    # pyautogui.click(actions.TOP_LEFT_X + 10, actions.TOP_LEFT_Y + 10)
    # time.sleep(0.2)
    # actions.detect_match_over()
    print(actions.click_battle_start())
    # actions.detect_game_end()
    # import cv2
    # import numpy as np
    # print(actions.detect_game_end())
    # Load images
    # large_image = cv2.imread(r'C:\Users\shour\RL\CLASH_ROYALE\images\scr3.jpg')
    # template = cv2.imread(r'C:\Users\shour\RL\CLASH_ROYALE\check_images\image.png')
    # confidence = 0.7
    # img = os.path.join(r'C:\Users\shour\RL\CLASH_ROYALE\check_images', "image.png")
    # location = pyautogui.locateOnScreen(
    #                     img,
    #                     confidence=confidence,
    #                     grayscale=True,
    #                     region=(win.left,win.top,win.width,win.height)
    # )
    # x, y = pyautogui.center(location)
    # print("coordinates",x,y)
    # Convert to grayscale

    # Perform template matching
    # result = cv2.matchTemplate(large_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find the best match location
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Define threshold for good match (e.g., 80%)
    # threshold = 0.8
    # if max_val >= threshold:
    #     h, w = template_gray.shape
    #     top_left = max_loc
    #     bottom_right = (top_left[0] + w, top_left[1] + h)
    #     cv2.rectangle(large_image, top_left, bottom_right, (0, 255, 0), 2)
    #     print("Match found!")
    # else:
    #     print("No match detected.")

    # # Display result
    # cv2.imshow('Detected', large_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()   
    # actions.card_play(1367,600,0)




    # val = pyautogui.center()
    # screenshot = pyautogui.screenshot(region=(
    #         win.left + int(0.1 * win.width),
    #         win.top + int(0.3 * win.height),
    #         int(0.8 * win.width),
    #         int(0.17 * win.height)
    # ))
    # screenshot.save(r"C:\Users\shour\RL\CLASH_ROYALE\images\scr3.jpg")
        # region = (
        #     self.TOP_LEFT_X + int(0.31 * self.WIDTH),
        #     self.TOP_LEFT_Y + int(0.73 * self.HEIGHT),
        #     int(0.32 * self.WIDTH),
        #     int(0.12 * self.HEIGHT)
        # )




    # print("Elixir: ",actions.count_elixir())
    # actions.card_play(win.left  + 100, 450, 2)
    # action = Actions(0,0,0,0)
    # dic = [
    # {'class': 'knight', 'x': 2},
    # {'class': 'archer', 'x': 0},
    # {'class': 'ar', 'x': 1},
    # {'class': 'her', 'x': 3}
    # ]
    # print(action.current_card_positions)
    # action.update_card_positions(dic)
    # print(action.current_card_positions)
    # dic1 = [
    # {'class': 'arpit', 'x': 3},
    # {'class': 'narpt', 'x': 2},
    # {'class': 'shr', 'x': 0},
    # {'class': 'her', 'x': 1}
    # ]
    # action.update_card_positions(dic1)
    # print(action.current_card_positions)
    print(5)
