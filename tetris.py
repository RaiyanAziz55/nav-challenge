"""
    The classic Tetris game developed using PyGame.
    Copyright (C) 2018 Recursos Python - recursospython.com.
    Copyright (C) 2021 Python Assets - pythonassets.com.
"""

from multiprocessing import Process
from collections import OrderedDict
import random

from pygame import Rect
import pygame
import numpy as np
<<<<<<< Updated upstream


WINDOW_WIDTH, WINDOW_HEIGHT = 500, 601
GRID_WIDTH, GRID_HEIGHT = 300, 600
TILE_SIZE = 30
=======
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
        self.reset()
        self.held_piece = None
        self.hold_flag = False
        
        # Initialize the AI agent
        self.agent = DQNAgent(state_size=self.get_state_size(), modelFile="sample.keras")
>>>>>>> Stashed changes


def remove_empty_columns(arr, _x_offset=0, _keep_counting=True):
    """
    Remove empty columns from arr (i.e. those filled with zeros).
    The return value is (new_arr, x_offset), where x_offset is how
    much the x coordinate needs to be increased in order to maintain
    the block's original position.
    """
    for colid, col in enumerate(arr.T):
        if col.max() == 0:
            if _keep_counting:
                _x_offset += 1
            # Remove the current column and try again.
            arr, _x_offset = remove_empty_columns(
                np.delete(arr, colid, 1), _x_offset, _keep_counting)
            break
        else:
            _keep_counting = False
    return arr, _x_offset


class BottomReached(Exception):
    pass


class TopReached(Exception):
    pass


class Block(pygame.sprite.Sprite):
    
<<<<<<< Updated upstream
    @staticmethod
    def collide(block, group):
        """
        Check if the specified block collides with some other block
        in the group.
        """
        for other_block in group:
            # Ignore the current block which will always collide with itself.
            if block == other_block:
                continue
            if pygame.sprite.collide_mask(block, other_block) is not None:
=======
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
        '''Draw the AI's recommended move on the board'''
        recommended_action = get_ai_recommendation(self, self.agent)
        if not recommended_action:
            return  # No recommendation available
        
        x, rotation = recommended_action
        simulated_piece = Tetris.TETROMINOS[self.current_piece][rotation]
        simulated_piece = [np.add(p, [x, 0]) for p in simulated_piece]

        # Simulate the drop
        while not self._check_collision(simulated_piece, [x, 0]):
            for p in simulated_piece:
                p[1] += 1
        for p in simulated_piece:
            p[1] -= 1

        # Draw the recommended position as green blocks
        for x, y in simulated_piece:
            if 0 <= y < Tetris.BOARD_HEIGHT and 0 <= x < Tetris.BOARD_WIDTH:
                self.board[y][x] = 3  # Use green color for AI recommendation

    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
        return board


    def get_game_score(self):
        '''Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
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
>>>>>>> Stashed changes
                return True
        return False
    
    def __init__(self):
        super().__init__()
        # Get a random color.
        self.color = random.choice((
            (200, 200, 200),
            (215, 133, 133),
            (30, 145, 255),
            (0, 170, 0),
            (180, 0, 140),
            (200, 200, 0)
        ))
        self.current = True
        self.struct = np.array(self.struct)
        # Initial random rotation and flip.
        if random.randint(0, 1):
            self.struct = np.rot90(self.struct)
        if random.randint(0, 1):
            # Flip in the X axis.
            self.struct = np.flip(self.struct, 0)
        self._draw()
    
    def _draw(self, x=4, y=0):
        width = len(self.struct[0]) * TILE_SIZE
        height = len(self.struct) * TILE_SIZE
        self.image = pygame.surface.Surface([width, height])
        self.image.set_colorkey((0, 0, 0))
        # Position and size
        self.rect = Rect(0, 0, width, height)
        self.x = x
        self.y = y
        for y, row in enumerate(self.struct):
            for x, col in enumerate(row):
                if col:
                    pygame.draw.rect(
                        self.image,
                        self.color,
                        Rect(x*TILE_SIZE + 1, y*TILE_SIZE + 1,
                             TILE_SIZE - 2, TILE_SIZE - 2)
                    )
        self._create_mask()
    
    def redraw(self):
        self._draw(self.x, self.y)
    
    def _create_mask(self):
        """
        Create the mask attribute from the main surface.
        The mask is required to check collisions. This should be called
        after the surface is created or update.
        """
        self.mask = pygame.mask.from_surface(self.image)
    
    def initial_draw(self):
        raise NotImplementedError
    
    @property
    def group(self):
        return self.groups()[0]
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value
        self.rect.left = value*TILE_SIZE
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = value
        self.rect.top = value*TILE_SIZE
    
    def move_left(self, group):
        self.x -= 1
        # Check if we reached the left margin.
        if self.x < 0 or Block.collide(self, group):
            self.x += 1
    
    def move_right(self, group):
        self.x += 1
        # Check if we reached the right margin or collided with another
        # block.
        if self.rect.right > GRID_WIDTH or Block.collide(self, group):
            # Rollback.
            self.x -= 1
    
    def move_down(self, group):
        self.y += 1
        # Check if the block reached the bottom or collided with 
        # another one.
        if self.rect.bottom > GRID_HEIGHT or Block.collide(self, group):
            # Rollback to the previous position.
            self.y -= 1
            self.current = False
            raise BottomReached
    
    def rotate(self, group):
        self.image = pygame.transform.rotate(self.image, 90)
        # Once rotated we need to update the size and position.
        self.rect.width = self.image.get_width()
        self.rect.height = self.image.get_height()
        self._create_mask()
        # Check the new position doesn't exceed the limits or collide
        # with other blocks and adjust it if necessary.
        while self.rect.right > GRID_WIDTH:
            self.x -= 1
        while self.rect.left < 0:
            self.x += 1
        while self.rect.bottom > GRID_HEIGHT:
            self.y -= 1
        while True:
            if not Block.collide(self, group):
                break
            self.y -= 1
        self.struct = np.rot90(self.struct)
    
    def update(self):
        if self.current:
            self.move_down()

    def predict_landing_position(self, group):
        """Predict the landing position of the block."""
        original_y = self.y
        while True:
            self.y += 1
            if self.rect.bottom > GRID_HEIGHT or Block.collide(self, group):
                # Rollback to the last valid position.
                self.y -= 1
                break
        landing_position = self.y
        self.y = original_y  # Restore original position
        return landing_position

    def draw_shadow(self, screen, group):
        landing_position = self.predict_landing_position(group)
        original_y = self.y
        self.y = landing_position
        for y, row in enumerate(self.struct):
            for x, col in enumerate(row):
                if col:
                    pygame.draw.rect(
                        screen,
                        (self.color[0] // 2, self.color[1] // 2, self.color[2] // 2),  # Darker color
                        Rect((self.x + x) * TILE_SIZE + 1, (self.y + y) * TILE_SIZE + 1,
                            TILE_SIZE - 2, TILE_SIZE - 2)
                    )
        self.y = original_y  # Restore original position


class SquareBlock(Block):
    struct = (
        (1, 1),
        (1, 1)
    )


class TBlock(Block):
    struct = (
        (1, 1, 1),
        (0, 1, 0)
    )


class LineBlock(Block):
    struct = (
        (1,),
        (1,),
        (1,),
        (1,)
    )


class LBlock(Block):
    struct = (
        (1, 1),
        (1, 0),
        (1, 0),
    )


class ZBlock(Block):
    struct = (
        (0, 1),
        (1, 1),
        (1, 0),
    )


class BlocksGroup(pygame.sprite.OrderedUpdates):
    
    @staticmethod
    def get_random_block():
        return random.choice(
            (SquareBlock, TBlock, LineBlock, LBlock, ZBlock))()
    
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self._reset_grid()
        self._ignore_next_stop = False
        self.score = 0
        self.next_block = None
        # Not really moving, just to initialize the attribute.
        self.stop_moving_current_block()
        # The first block.
        self._create_new_block()
    
    def _check_line_completion(self):
        """
        Check each line of the grid and remove the ones that
        are complete.
        """
        # Start checking from the bottom.
        for i, row in enumerate(self.grid[::-1]):
            if all(row):
                self.score += 5
                # Get the blocks affected by the line deletion and
                # remove duplicates.
                affected_blocks = list(
                    OrderedDict.fromkeys(self.grid[-1 - i]))
                
                for block, y_offset in affected_blocks:
                    # Remove the block tiles which belong to the
                    # completed line.
                    block.struct = np.delete(block.struct, y_offset, 0)
                    if block.struct.any():
                        # Once removed, check if we have empty columns
                        # since they need to be dropped.
                        block.struct, x_offset = \
                            remove_empty_columns(block.struct)
                        # Compensate the space gone with the columns to
                        # keep the block's original position.
                        block.x += x_offset
                        # Force update.
                        block.redraw()
                    else:
                        # If the struct is empty then the block is gone.
                        self.remove(block)
                
                # Instead of checking which blocks need to be moved
                # once a line was completed, just try to move all of
                # them.
                for block in self:
                    # Except the current block.
                    if block.current:
                        continue
                    # Pull down each block until it reaches the
                    # bottom or collides with another block.
                    while True:
                        try:
                            block.move_down(self)
                        except BottomReached:
                            break
                
                self.update_grid()
                # Since we've updated the grid, now the i counter
                # is no longer valid, so call the function again
                # to check if there're other completed lines in the
                # new grid.
                self._check_line_completion()
                break
    
    def _reset_grid(self):
        self.grid = [[0 for _ in range(10)] for _ in range(20)]
    
    def _create_new_block(self):
        new_block = self.next_block or BlocksGroup.get_random_block()
        if Block.collide(new_block, self):
            raise TopReached
        self.add(new_block)
        self.next_block = BlocksGroup.get_random_block()
        self.update_grid()
        self._check_line_completion()
    
    def update_grid(self):
        self._reset_grid()
        for block in self:
            for y_offset, row in enumerate(block.struct):
                for x_offset, digit in enumerate(row):
                    # Prevent replacing previous blocks.
                    if digit == 0:
                        continue
                    rowid = block.y + y_offset
                    colid = block.x + x_offset
                    self.grid[rowid][colid] = (block, y_offset)
    
    @property
    def current_block(self):
        return self.sprites()[-1]
    
    def update_current_block(self):
        try:
            self.current_block.move_down(self)
        except BottomReached:
            self.stop_moving_current_block()
            self._create_new_block()
        else:
<<<<<<< Updated upstream
            self.update_grid()
    
    def move_current_block(self):
        # First check if there's something to move.
        if self._current_block_movement_heading is None:
            return
        action = {
            pygame.K_DOWN: self.current_block.move_down,
            pygame.K_LEFT: self.current_block.move_left,
            pygame.K_RIGHT: self.current_block.move_right
        }
        try:
            # Each function requires the group as the first argument
            # to check any possible collision.
            action[self._current_block_movement_heading](self)
        except BottomReached:
            self.stop_moving_current_block()
            self._create_new_block()
=======
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
        if key == ord('h'):  # Press 'H' to use AI recommendation
            recommendation = get_ai_recommendation(self, self.agent)
            if recommendation:
                self.current_pos[0], self.current_rotation = recommendation
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
>>>>>>> Stashed changes
        else:
            self.update_grid()
    
    def start_moving_current_block(self, key):
        if self._current_block_movement_heading is not None:
            self._ignore_next_stop = True
        self._current_block_movement_heading = key
    
    def stop_moving_current_block(self):
        if self._ignore_next_stop:
            self._ignore_next_stop = False
        else:
            self._current_block_movement_heading = None
    
    def rotate_current_block(self):
        # Prevent SquareBlocks rotation.
        if not isinstance(self.current_block, SquareBlock):
            self.current_block.rotate(self)
            self.update_grid()

    def pause_game(self):
        """Pause the game."""
        paused = True
        while paused:
            # Render the current screen and overlay "Game Paused" text
            canvas = self.get_canvas()  # Create the current canvas (game board + menu)
            cv2.putText(canvas, "Game Paused", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            # Display the updated canvas
            cv2.imshow('Tetris', canvas)
            key = cv2.waitKey(100)

            # Handle unpause or quit during pause
            if key == ord('p'):  # Unpause the game
                paused = False
            elif key == ord('q'):  # Quit the game
                self.game_over = True
                paused = False



def draw_grid(background):
    """Draw the background grid."""
    grid_color = 50, 50, 50
    # Vertical lines.
    for i in range(11):
        x = TILE_SIZE * i
        pygame.draw.line(
            background, grid_color, (x, 0), (x, GRID_HEIGHT)
        )
    # Horizontal liens.
    for i in range(21):
        y = TILE_SIZE * i
        pygame.draw.line(
            background, grid_color, (0, y), (GRID_WIDTH, y)
        )


def draw_centered_surface(screen, surface, y):
    screen.blit(surface, (400 - surface.get_width()//2, y))
def main_menu():
    pygame.init()
    pygame.display.set_caption("Tetris with PyGame - Menu")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    # Colors
    white = (255, 255, 255)
    black = (0, 0, 0)
    blue = (30, 145, 255)

    # Fonts
    try:
        font = pygame.font.Font("Roboto-Regular.ttf", 30)
    except OSError:
        font = pygame.font.Font(pygame.font.get_default_font(), 30)
    intro_font = pygame.font.Font(pygame.font.get_default_font(), 40)

    # Text
    intro_text = intro_font.render("Welcome to Tetris!", True, blue)
    singleplay_text = font.render("SINGLEPLAYER", True, white)
    multiplay_text = font.render("MULTIPLAYER", True, white)

    # Rectangles for buttons
    singleplay_rect = singleplay_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
    multiplay_rect = multiplay_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))

    # Menu loop
    run = True
    selected_option = None
    while run:
        screen.fill(black)

        # Draw intro text
        screen.blit(intro_text, (WINDOW_WIDTH // 2 - intro_text.get_width() // 2, 150))

        # Draw the "Singleplayer" button
        pygame.draw.rect(screen, blue, singleplay_rect.inflate(20, 10))
        screen.blit(singleplay_text, singleplay_rect)

        # Draw the "Multiplayer" button
        pygame.draw.rect(screen, blue, multiplay_rect.inflate(20, 10))
        screen.blit(multiplay_text, multiplay_rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if singleplay_rect.collidepoint(event.pos):
                    selected_option = "singleplayer"
                    run = False
                elif multiplay_rect.collidepoint(event.pos):
                    selected_option = "multiplayer"
                    run = False

        pygame.display.flip()
        clock.tick(60)

    return selected_option
        
def difficulty_menu():
    pygame.init()
    pygame.display.set_caption("Tetris with PyGame - Select Difficulty")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    # Colors
    white = (255, 255, 255)
    black = (0, 0, 0)
    blue = (30, 145, 255)

    # Fonts
    try:
        font = pygame.font.Font("Roboto-Regular.ttf", 30)
    except OSError:
        font = pygame.font.Font(pygame.font.get_default_font(), 30)
    intro_font = pygame.font.Font(pygame.font.get_default_font(), 40)

     # Text
    intro_text = intro_font.render("Select your difficulty:", True, blue)
    easy_text = font.render("EASY", True, white)
    medium_text = font.render("MEDIUM", True, white)
    hard_text = font.render("HARD", True, white)

    # Rectangles for buttons
    easy_rect = easy_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
    medium_rect = medium_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
    hard_rect = hard_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))

    # Menu loop
    run = True
    selected_difficulty = None
    while run:
        screen.fill(black)

        # Draw intro text
        screen.blit(intro_text, (WINDOW_WIDTH // 2 - intro_text.get_width() // 2, 150))

        # Draw buttons
        pygame.draw.rect(screen, blue, easy_rect.inflate(20, 10))
        screen.blit(easy_text, easy_rect)

        pygame.draw.rect(screen, blue, medium_rect.inflate(20, 10))
        screen.blit(medium_text, medium_rect)

        pygame.draw.rect(screen, blue, hard_rect.inflate(20, 10))
        screen.blit(hard_text, hard_rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if easy_rect.collidepoint(event.pos):
                    selected_difficulty = 'easy'
                    run = False
                elif medium_rect.collidepoint(event.pos):
                    selected_difficulty = 'medium'
                    run = False
                elif hard_rect.collidepoint(event.pos):
                    selected_difficulty = 'hard'
                    run = False

        pygame.display.flip()
        #clock.tick(60)
        
    return selected_difficulty


def run_game_instance(difficulty, instance_id):
    pygame.init()
    pygame.display.set_caption("Tetris with PyGame")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    run = True
    paused = False
    game_over = False
    # Create background.
    background = pygame.Surface(screen.get_size())
    bgcolor = (0, 0, 0)
    background.fill(bgcolor)
    # Draw the grid on top of the background.
    draw_grid(background)
    # This makes blitting faster.
    background = background.convert()
    
    try:
        font = pygame.font.Font("Roboto-Regular.ttf", 20)
    except OSError:
        # If the font file is not available, the default will be used.
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
    next_block_text = font.render(
        "Next figure:", True, (255, 255, 255), bgcolor)
    score_msg_text = font.render(
        "Score:", True, (255, 255, 255), bgcolor)
    game_over_text = font.render(
        "¡Game over!", True, (255, 220, 0), bgcolor)
    
    #Difficulty Settings
    DIFFICULTY_SETTINGS = {
    'easy': {
        'BLOCK_SPEED': 750,            # Falling speed (ms)
        #'shape_variability': 2,   # Maximum number of block types
        #'move_window': 500,       # Time for lateral movement (ms)
        #'rotations_allowed': 4,   # Unlimited rotations
        #'death_height': 20        # Maximum rows allowed
    },
    'medium': {
        'BLOCK_SPEED': 375,
      
    },
    'hard': {
        'BLOCK_SPEED': 150,
      
    }
}

    
    # Event constants.
    MOVEMENT_KEYS = pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN
    EVENT_UPDATE_CURRENT_BLOCK = pygame.USEREVENT + 1
    EVENT_MOVE_CURRENT_BLOCK = pygame.USEREVENT + 2
    pygame.time.set_timer(EVENT_UPDATE_CURRENT_BLOCK, DIFFICULTY_SETTINGS[difficulty]['BLOCK_SPEED'])
    pygame.time.set_timer(EVENT_MOVE_CURRENT_BLOCK, 100)
    
    blocks = BlocksGroup()
    
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            elif event.type == pygame.KEYUP:
                if not paused and not game_over:
                    if event.key in MOVEMENT_KEYS:
                        blocks.stop_moving_current_block()
                    elif event.key == pygame.K_UP:
                        blocks.rotate_current_block()
                if event.key == pygame.K_p:
                    paused = not paused
            
            # Stop moving blocks if the game is over or paused.
            if game_over or paused:
                continue
            
            if event.type == pygame.KEYDOWN:
                if event.key in MOVEMENT_KEYS:
                    blocks.start_moving_current_block(event.key)
            
            try:
                if event.type == EVENT_UPDATE_CURRENT_BLOCK:
                    blocks.update_current_block()
                elif event.type == EVENT_MOVE_CURRENT_BLOCK:
                    blocks.move_current_block()
            except TopReached:
                game_over = True
        
        # Draw background and grid.
        screen.blit(background, (0, 0))
        # Blocks.
        blocks.draw(screen)
        # Draw shadow block before the current block.
        blocks.current_block.draw_shadow(screen, blocks)
        # Draw current blocks.
        blocks.draw(screen)

        # Sidebar with misc. information.
        draw_centered_surface(screen, next_block_text, 50)
        draw_centered_surface(screen, blocks.next_block.image, 100)
        draw_centered_surface(screen, score_msg_text, 240)
        score_text = font.render(
            str(blocks.score), True, (255, 255, 255), bgcolor)
        draw_centered_surface(screen, score_text, 270)
        if game_over:
            draw_centered_surface(screen, game_over_text, 360)
        # Update.
        pygame.display.flip()
        
        
def run_multi_instance(difficulty):
    """Run two game instances using multi-processing."""
    process1 = Process(target=run_game_instance, args=(difficulty, 1))
    process2 = Process(target=run_game_instance, args=(difficulty, 2))     

    # Start both processes
    process1.start()
    process2.start()

    # Wait for both processes to finish
    process1.join()
    process2.join()

<<<<<<< Updated upstream


def main():
    
    selected_option = main_menu()
    difficulty = difficulty_menu()

    run_game_instance(difficulty,1)
    
    
    pygame.quit()
=======
    def render(self):
        """Display the current canvas."""
        canvas = self.get_canvas()  # Get the complete canvas
        cv2.imshow('Tetris', canvas)
        cv2.waitKey(1)



def main():
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

>>>>>>> Stashed changes


if __name__ == "__main__":
    main()
