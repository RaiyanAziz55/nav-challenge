import random
import cv2
import numpy as np
from PIL import Image
from time import sleep
from run_model import get_ai_recommendation
from dqn_agent import DQNAgent
from menu import Menu  # Import the Menu class

# Tetris game class
class Tetris:

    '''Tetris game class'''

    # BOARD
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: { # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # S
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # O
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
        3: (0, 255, 111),
    }

    def __init__(self):
        self.agent = DQNAgent(state_size=self.get_state_size(), modelFile="sample.keras")
        self.reset()
        self.held_piece = None
        self.hold_flag = False
        self.show_recommendation = False
    

    def reset(self):
        '''Resets the game, returning the current state'''
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        self.score = 0
        return self._get_board_props(self.board)

    def update_game_state(self):
        """Automatically update the game state by moving the current piece down."""
        # Try to move the piece down
        self.current_pos[1] += 1
        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            # If collision occurs, revert the move and finalize the piece
            self.current_pos[1] -= 1
            self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
            lines_cleared, self.board = self._clear_lines(self.board)
            self.score += lines_cleared * 10  # Update score based on lines cleared

            # Start a new round
            self._new_round()

            # Check for game over
            if self._check_collision(self._get_rotated_piece(), self.current_pos):
                self.game_over = True

    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]

    def draw_ai_recommendation(self):
        '''Calculate the AI's recommended move and return the coordinates.'''
        recommended_action = get_ai_recommendation(self, self.agent)
        if not recommended_action:
            return []  # No recommendation available

        x, rotation = recommended_action
        simulated_piece = Tetris.TETROMINOS[self.current_piece][rotation]
        simulated_piece = [np.add(p, [x, 0]) for p in simulated_piece]

        # Simulate the drop
        while not self._check_collision(simulated_piece, [0, 0]):
            for p in simulated_piece:
                p[1] += 1  # Move piece down
        for p in simulated_piece:
            p[1] -= 1  # Adjust after collision

        # Return the final position of the recommendation
        return [(int(px), int(py)) for px, py in simulated_piece if 0 <= py < Tetris.BOARD_HEIGHT and 0 <= px < Tetris.BOARD_WIDTH]

    def apply_ai_recommendation(self):
        """Move the current piece to the AI-recommended position."""
        if not self.reccomendantion:
            print("AI recommendation is missing or invalid.")
            return

        # Extract the recommended position
        x, rotation = get_ai_recommendation(self, self.agent)
        print(f"Applying AI Recommendation: x={x}, rotation={rotation}")

        # Apply the recommendation
        self.current_pos[0] = x  # Update the horizontal position
        self.current_rotation = rotation  # Update the rotation

        # Perform a hard drop to finalize the position
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.current_pos[1] += 1
        self.current_pos[1] -= 1  # Adjust after collision

        # Place the piece on the board
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)

        # Clear lines and update the game state
        lines_cleared, self.board = self._clear_lines(self.board)
        self.score += lines_cleared * 10  # Update score based on lines cleared

        # Start a new round
        self._new_round()



    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
        return board

    def get_game_score(self):
        '''Returns the current game score.'''
        return self.score
    
    def _new_round(self):
        """Start a new round with a new piece."""
        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)

        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0
        self.hold_flag = False  # Allow holding again in the new turn

        # Compute AI recommendation
        self.reccomendantion = self.draw_ai_recommendation()

        # Check for game over
        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True


    def _check_collision(self, piece, pos):
        '''Check if there is a collision between the current piece and the board'''
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] == Tetris.MAP_BLOCK:
                return True
        return False

    def _rotate(self, angle):
        '''Change the current rotation'''
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r

    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board, returning the resulting board'''        
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = Tetris.MAP_BLOCK
        return board

    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == Tetris.BOARD_WIDTH]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board

    def _number_of_holes(self, board):
        '''Number of holes in the board (empty square with at least one block above it)'''
        holes = 0

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            holes += len([x for x in col[i+1:] if x == Tetris.MAP_EMPTY])

        return holes

    def _bumpiness(self, board):
        '''Sum of the differences of heights between pair of columns'''
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            min_ys.append(i)
        
        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i+1])

        return total_bumpiness, max_bumpiness

    def _height(self, board):
        '''Sum and maximum height of the board'''
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == Tetris.MAP_EMPTY:
                i += 1
            height = Tetris.BOARD_HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height

    def _get_board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]

    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_piece
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states

    def get_canvas(self):
        """Create the complete canvas (game board + menu)."""
        # Render game board
        img_board = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img_board = np.array(img_board).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img_board = img_board[..., ::-1]  # Convert RGB to BGR (used by cv2)
        img_board = Image.fromarray(img_board, 'RGB')
        img_board = img_board.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25), Image.NEAREST)
        img_board = np.array(img_board)

        # Create a blank canvas for the full display (game + menu)
        canvas_height = Tetris.BOARD_HEIGHT * 25
        canvas_width = (Tetris.BOARD_WIDTH * 25) + 200  # Add space for menu (e.g., 200px)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Paste the game board onto the left side of the canvas
        canvas[:, :img_board.shape[1]] = img_board

        # Draw the menu on the right side
        menu_x_start = img_board.shape[1] + 10  # Offset to the right of the game board
        cv2.putText(canvas, "Menu:", (menu_x_start, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, f"Score: {self.score}", (menu_x_start, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Options:", (menu_x_start, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, " - P: Pause", (menu_x_start, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, " - Q: Quit", (menu_x_start, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return canvas

    def handle_player_input(self, key):
        """Handle player key inputs."""
        if key == ord('h'):  # Press 'H' to toggle AI recommendation
            self.show_recommendation = not self.show_recommendation  # Toggle visibility
            return

        if key == ord('i'):  # Press 'I' to move piece to the AI-recommended position
            if self.reccomendantion:
                # Move the current piece to the recommended position
                self.apply_ai_recommendation()
            else:
                print("No valid AI recommendation available.")
            return

        # Handle normal player controls
        if key == ord('a'):  # Move left
            self.move_piece_left()
        elif key == ord('d'):  # Move right
            self.move_piece_right()
        elif key == ord('s'):  # Soft drop
            self.soft_drop()
        elif key == ord('w'):  # Hard drop
            self.hard_drop()
        elif key == ord("r"):  # Rotate piece
            self.rotate_piece(90)
        elif key == ord('p'):  # Pause game
            self.pause_game()
        elif key == ord('q'):  # Quit game
            self.game_over = True



    def get_state_size(self):
        '''Size of the state'''
        return 4

    def move_piece_left(self):
        """Move the current piece one step to the left."""
        new_pos = [self.current_pos[0] - 1, self.current_pos[1]]
        if not self._check_collision(self._get_rotated_piece(), new_pos):
            self.current_pos = new_pos

    def move_piece_right(self):
        """Move the current piece one step to the right."""
        new_pos = [self.current_pos[0] + 1, self.current_pos[1]]
        if not self._check_collision(self._get_rotated_piece(), new_pos):
            self.current_pos = new_pos

    def soft_drop(self):
        """Move the current piece down one step."""
        new_pos = [self.current_pos[0], self.current_pos[1] + 1]
        if not self._check_collision(self._get_rotated_piece(), new_pos):
            self.current_pos = new_pos

    def hard_drop(self):
        """Drop the current piece to the lowest valid position."""
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.current_pos[1] += 1
        self.current_pos[1] -= 1  # Adjust after collision

    def rotate_piece(self, angle):
        """Rotate the current piece."""
        new_rotation = (self.current_rotation + angle) % 360
        rotated_piece = Tetris.TETROMINOS[self.current_piece][new_rotation]
        if not self._check_collision(rotated_piece, self.current_pos):
            self.current_rotation = new_rotation

    def hold_piece(self):
        """Swap the current piece with the held piece."""
        if self.hold_flag:  # Prevent multiple holds in one turn
            return  # Do nothing if the player already used hold in this turn

        if self.held_piece is None:
            # First-time holding: store the current piece and start a new round
            self.held_piece = self.current_piece
            self._new_round()
        else:
            # Swap the current piece with the held piece
            self.current_piece, self.held_piece = self.held_piece, self.current_piece
            self.current_pos = [3, 0]  # Reset position for the new piece
            self.current_rotation = 0  # Reset rotation for the new piece

        self.hold_flag = True  # Disable further holds in this turn

    def pause_game(self):
        """Pause the game."""
        paused = True
        while paused:
            # Render the current screen and overlay "Game Paused" text
            canvas = self.get_canvas()  # Create the current canvas (game board + menu)
            cv2.putText(canvas, "Game Paused", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Display the updated canvas
            cv2.imshow('Tetris', canvas)
            key = cv2.waitKey(100)

            # Handle unpause or quit during pause
            if key == ord('p'):  # Unpause the game
                paused = False
            elif key == ord('q'):  # Quit the game
                self.game_over = True
                paused = False

    def play(self):
        while not self.game_over:
            # Render the game
            self.render()

            # Get key press
            key = cv2.waitKey(100)

            # Handle player input
            if key != -1:  # Check if a key was pressed
                self.handle_player_input(key)

            # Update game state (e.g., move pieces down automatically)
            self.update_game_state()

    def render(self):
        '''Render the current board along with the AI recommendation.'''
        # Get the current game board
        img_board = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img_board = np.array(img_board).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img_board = img_board[..., ::-1]  # Convert RGB to BGR (used by cv2)
        img_board = Image.fromarray(img_board, 'RGB')
        img_board = img_board.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25), Image.NEAREST)
        img_board = np.array(img_board)



    

        # Create a blank canvas for the full display (game + menu)
        canvas_height = Tetris.BOARD_HEIGHT * 25
        canvas_width = (Tetris.BOARD_WIDTH * 25) + 200  # Add space for menu (e.g., 200px)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Paste the game board onto the left side of the canvas
        canvas[:, :img_board.shape[1]] = img_board

           # Draw AI recommendation if toggled
        if self.show_recommendation and self.reccomendantion:
            for x, y in self.reccomendantion:
                top_left = (x * 25, y * 25)
                bottom_right = ((x + 1) * 25, (y + 1) * 25)
                cv2.rectangle(
                    canvas,
                    top_left,
                    bottom_right,
                    (0, 255, 111),  # Green for recommendation
                    -1
                )

        # Draw the menu on the right side
        menu_x_start = img_board.shape[1] + 10  # Offset to the right of the game board
        cv2.putText(canvas, "Menu:", (menu_x_start, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, f"Score: {self.score}", (menu_x_start, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Options:", (menu_x_start, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, " - P: Pause", (menu_x_start, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, " - Q: Quit", (menu_x_start, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the final canvas
        cv2.imshow('Tetris', canvas)
        cv2.waitKey(1)


def main():
    while True:  # Loop to allow restarting the game
        # Show the menu
        menu = Menu()
        choice = menu.show()

        if choice == "Play Game":
            game = Tetris()
            game.play()
        elif choice == "Set Difficulty":
            print("Difficulty selection not implemented yet!")
            # Add your difficulty logic here
        elif choice == "Quit":
            print("Exiting game...")
            break  # Exit the loop and quit the game


if __name__ == "__main__":
    main()