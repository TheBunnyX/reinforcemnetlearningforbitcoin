import pygame
import sys

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PLAYER_SIZE = 50
BULLET_SIZE = 10
BULLET_SPEED = 10
PLAYER_SPEED = 5

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("First Strike Game")

# Player positions and settings
player1 = pygame.Rect(50, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)
player2 = pygame.Rect(WIDTH - 100, HEIGHT // 2, PLAYER_SIZE, PLAYER_SIZE)
player1_bullets = []
player2_bullets = []
player1_direction = 1  # 1 = Right, -1 = Left
player2_direction = -1

# Game variables
game_over = False
winner = None

# Main game loop
while not game_over:
    screen.fill(WHITE)
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    # Key handling for players
    keys = pygame.key.get_pressed()
    
    # Player 1 controls (WASD)
    if keys[pygame.K_w] and player1.top > 0:
        player1.y -= PLAYER_SPEED
    if keys[pygame.K_s] and player1.bottom < HEIGHT:
        player1.y += PLAYER_SPEED
    if keys[pygame.K_a] and player1.left > 0:
        player1.x -= PLAYER_SPEED
        player1_direction = -1
    if keys[pygame.K_d] and player1.right < WIDTH:
        player1.x += PLAYER_SPEED
        player1_direction = 1
    if keys[pygame.K_SPACE]:  # Player 1 fires bullet
        bullet = pygame.Rect(player1.centerx, player1.centery, BULLET_SIZE, BULLET_SIZE)
        player1_bullets.append((bullet, player1_direction))
    
    # Player 2 controls (Arrow keys)
    if keys[pygame.K_UP] and player2.top > 0:
        player2.y -= PLAYER_SPEED
    if keys[pygame.K_DOWN] and player2.bottom < HEIGHT:
        player2.y += PLAYER_SPEED
    if keys[pygame.K_LEFT] and player2.left > 0:
        player2.x -= PLAYER_SPEED
        player2_direction = -1
    if keys[pygame.K_RIGHT] and player2.right < WIDTH:
        player2.x += PLAYER_SPEED
        player2_direction = 1
    if keys[pygame.K_RETURN]:  # Player 2 fires bullet
        bullet = pygame.Rect(player2.centerx, player2.centery, BULLET_SIZE, BULLET_SIZE)
        player2_bullets.append((bullet, player2_direction))
    
    # Move bullets
    for bullet, direction in player1_bullets:
        bullet.x += BULLET_SPEED * direction
        if bullet.colliderect(player2):
            winner = "Player 1"
            game_over = True
    for bullet, direction in player2_bullets:
        bullet.x += BULLET_SPEED * direction
        if bullet.colliderect(player1):
            winner = "Player 2"
            game_over = True
    
    # Draw players
    pygame.draw.rect(screen, RED, player1)
    pygame.draw.rect(screen, BLUE, player2)
    
    # Draw bullets
    for bullet, _ in player1_bullets:
        pygame.draw.rect(screen, RED, bullet)
    for bullet, _ in player2_bullets:
        pygame.draw.rect(screen, BLUE, bullet)
    
    # Update screen
    pygame.display.flip()
    pygame.time.Clock().tick(60)

# Game Over
screen.fill(WHITE)
font = pygame.font.Font(None, 74)
text = font.render(f"{winner} Wins!", True, (0, 0, 0))
screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
pygame.display.flip()
pygame.time.delay(3000)
pygame.quit()

