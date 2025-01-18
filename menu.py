import cv2
import numpy as np

class Menu:
    def __init__(self):
        self.options = ["Play Game", "Set Difficulty", "Quit"]
        self.selected_index = 0

    def render(self):
        '''Display the menu'''
        img = np.zeros((400, 600, 3), dtype=np.uint8)  # Create a blank black screen
        for i, option in enumerate(self.options):
            color = (0, 255, 0) if i == self.selected_index else (255, 255, 255)
            cv2.putText(img, option, (50, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Menu", img)

    def handle_input(self, key):
        '''Handle key input to navigate menu'''
        if key == ord("w"):  # Move up
            self.selected_index = (self.selected_index - 1) % len(self.options)
        elif key == ord("s"):  # Move down
            self.selected_index = (self.selected_index + 1) % len(self.options)
        elif key == ord("e"):  # Select (Enter)
            return self.options[self.selected_index]
        return None

    def show(self):
        '''Main menu loop'''
        while True:
            self.render()
            key = cv2.waitKey(100)
            choice = self.handle_input(key)
            if choice:
                cv2.destroyAllWindows()
                return choice
