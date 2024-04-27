import pygame
import random

pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Set the width and height of the screen [width, height]
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
MARGIN = 1
GRID_WIDTH = SCREEN_WIDTH // BLOCK_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // BLOCK_SIZE

# Set up the grid
grid = []
for row in range(GRID_HEIGHT):
    grid.append([])
    for column in range(GRID_WIDTH):
        grid[row].append(0)

# Define the shapes of the single parts
tetris_shapes = [
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 2, 2],
     [2, 2, 0]],

    [[3, 3, 0],
     [0, 3, 3]],

    [[4, 0, 0],
     [4, 4, 4]],

    [[0, 0, 5],
     [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
     [7, 7]]
]

# Define the colors of the single parts
tetris_colors = [
    BLUE,
    GREEN,
    RED,
    WHITE,

]


# Define the class for the tetris piece
class TetrisPiece:
    def __init__(self, shape, color):
        self.shape = shape
        self.color = color
        self.x = GRID_WIDTH // 2 - len(shape[0]) // 2
        self.y = 0

    def rotate(self):
        self.shape = [[self.shape[y][x] for y in range(len(self.shape))] for x in range(len(self.shape[0]) - 1, -1, -1)]

    def move_down(self):
        self.y += 1

    def move_left(self):
        self.x -= 1

    def move_right(self):
        self.x += 1

    def draw(self):
        for y, row in enumerate(self.shape):
            for x, val in enumerate(row):
                if val > 0:
                    pygame.draw.rect(screen, self.color,
                     [(self.x + x) * BLOCK_SIZE + MARGIN, (self.y + y) * BLOCK_SIZE + MARGIN, BLOCK_SIZE - 2 * MARGIN,
                      BLOCK_SIZE - 2 * MARGIN])

def check_collision(piece, grid):
    for y, row in enumerate(piece.shape):
        for x, val in enumerate(row):
            if val > 0:
                if piece.y + y >= GRID_HEIGHT or piece.x + x < 0 or piece.x + x >= GRID_WIDTH or grid[piece.y + y][
                    piece.x + x] > 0:
                    return True
    return False

def merge_piece(piece, grid):
    for y, row in enumerate(piece.shape):
        for x, val in enumerate(row):
            if val > 0:
                grid[piece.y + y][piece.x + x] = piece.color

def check_lines(grid):
    lines_cleared = 0
    for y in range(GRID_HEIGHT):
        if all(grid[y]):
            lines_cleared += 1
            for i in range(y, 0, -1):
                grid[i] = grid[i - 1][:]
            grid[0] = [0] * GRID_WIDTH
    return lines_cleared

def draw_grid(grid):
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            if val > 0:
                pygame.draw.rect(screen, tetris_colors[val - 1],
                                 [(x) * BLOCK_SIZE + MARGIN, (y) * BLOCK_SIZE + MARGIN, BLOCK_SIZE - 2 * MARGIN,
                                  BLOCK_SIZE - 2 * MARGIN])

def draw_score(score):
    font = pygame.font.Font(None, 36)
    text = font.render("Score: " + str(score), True, WHITE)
    screen.blit(text, [10, 10])


if __name__ == '__main__':
     # Set up the game
    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
    pygame.display.set_caption("Tetris")

    # Set up the clock
    clock = pygame.time.Clock()

    # Set up the game variables
    fall_time = 0
    fall_speed = 0.5
    current_piece = TetrisPiece(random.choice(tetris_shapes), random.choice(tetris_colors))
    next_piece = TetrisPiece(random.choice(tetris_shapes), random.choice(tetris_colors))
    game_over = False
    score = 0

    while not game_over:
         # Handle events
         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 game_over = True
             elif event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_LEFT:
                     current_piece.move_left()
                     if check_collision(current_piece, grid):
                         current_piece.move_right()
                 elif event.key == pygame.K_RIGHT:
                     current_piece.move_right()
                     if check_collision(current_piece, grid):
                         current_piece.move_left()
                 elif event.key == pygame.K_DOWN:
                     current_piece.move_down()
                     if check_collision(current_piece, grid):
                         current_piece.move_up()
                 elif event.key == pygame.K_UP:
                     current_piece.rotate()
                     if check_collision(current_piece, grid):
                         current_piece.rotate()
         # Update the game
         current_time = pygame.time.get_ticks()
         if current_time - fall_time > fall_speed * 1000:
             fall_time = current_time
             current_piece.move_down()
             if check_collision(current_piece, grid):
                 current_piece.move_up()
                 merge_piece(current_piece, grid)
                 lines_cleared = check_lines(grid)
                 score += lines_cleared ** 2
                 current_piece = next_piece
                 next_piece = TetrisPiece(random.choice(tetris_shapes), random.choice(tetris_colors))
                 if check_collision(current_piece, grid):
                     game_over = True

         # Draw the game
         screen.fill(BLACK)
         draw_grid(grid)
         current_piece.draw(screen)
         draw_score(score)
         pygame.display.flip()

         # Set the game's FPS
         clock.tick(60)


    pygame.quit()

    # Define the functions for the game


