import pygame
import random

def initialize_game_variables():
    """Initialize game variables for character position, pipe position, and score."""
    global chr_x, chr_y, pipe_x, score
    chr_x, chr_y = 350, 250
    pipe_x = 800
    score = 0

# Initialize Pygame and game variables
pygame.init()
initialize_game_variables()

# Screen setup
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Character setup
chr_img = pygame.image.load('./assets/flappy-tomato-removebg.png')

# Pipe setup
pipe_height = 300
pipe_gap = 200
pipe_speed = 2

# Score setup
font = pygame.font.Font('freesansbold.ttf', 32)
score = 0

# Game Over setup
game_over_font = pygame.font.Font('freesansbold.ttf', 64)

# Restart Button setup
button_color = (0, 255, 0)
button_x, button_y, button_width, button_height = 300, 400, 200, 50

# Define action space
actions = [0, 1]

# Initialize reward
reward = 0

# Main game loop
running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen
    state = (chr_y, pipe_height - chr_y, pipe_x - chr_x)


    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                chr_y -= 50

    # Character logic
    screen.blit(chr_img, (chr_x, chr_y))
    chr_y += 1  # Gravity

    # Pipe logic
    pygame.draw.rect(screen, (0, 255, 0), (pipe_x, 0, 100, pipe_height))
    pygame.draw.rect(screen, (0, 255, 0), (pipe_x, pipe_height + pipe_gap, 100, SCREEN_HEIGHT))
    pipe_x -= pipe_speed
    if pipe_x < -100:
        pipe_x = 800
        pipe_height = random.randint(100, 300)  # Randomize pipe height

    # Collision detection
    chr_rect = pygame.Rect(chr_x, chr_y, 50, 50)
    upper_pipe_rect = pygame.Rect(pipe_x, 0, 100, pipe_height)
    lower_pipe_rect = pygame.Rect(pipe_x, pipe_height + pipe_gap, 100, SCREEN_HEIGHT)

    if chr_rect.colliderect(upper_pipe_rect) or chr_rect.colliderect(lower_pipe_rect):
        game_over_text = game_over_font.render('Game Over', True, (255, 0, 0))
        screen.blit(game_over_text, (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 3))
        pygame.draw.rect(screen, button_color, (button_x, button_y, button_width, button_height))
        restart_text = font.render('Restart', True, (0, 0, 0))
        screen.blit(restart_text, (button_x + 50, button_y + 10))
        pygame.display.update()
        
        waiting_for_restart = True
        while waiting_for_restart:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting_for_restart = False
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if button_x <= mouse_x <= button_x + button_width and button_y <= mouse_y <= button_y + button_height:
                        initialize_game_variables()
                        waiting_for_restart = False

    # Scoring
    score_display = font.render(f'Score: {score}', True, (255, 255, 255))
    screen.blit(score_display, (10, 10))
    if pipe_x + 100 < chr_x and pipe_x + 100 + pipe_speed >= chr_x:
        score += 1

    # Inside the main loop, update reward based on game events
    if chr_rect.colliderect(upper_pipe_rect) or chr_rect.colliderect(lower_pipe_rect):
        reward = -1
    elif pipe_x + 100 < chr_x and pipe_x + 100 + pipe_speed >= chr_x:
        reward = 1
    else:
        reward = 0  # No reward or penalty for other states

    pygame.display.update()
