import pygame
import random as rd


def draw_snake(screen: pygame.Surface, snake: tuple[int, int],
               radius: int, head_img: pygame.Surface,
               body_img: pygame.Surface) -> None:
    """Draw the snake on the screen, using a head image for
    the first node and body image for the rest."""
    for i, node in enumerate(snake):
        posX = (node[0] + 0.5) * 2 * radius
        posY = (node[1] + 0.5) * 2 * radius
        if i == 0:
            screen.blit(head_img, (posX, posY))
        else:
            screen.blit(body_img, (posX, posY))


def draw_item(screen: pygame.Surface, items: tuple[int, int],
              bad_items: tuple[int, int], radius: int,
              green_img: pygame.Surface, red_img: pygame.Surface) -> None:
    """Draw good items using green_img and bad items using red_img."""
    for i, item in enumerate(items):
        posX = (item[0] + 0.5) * 2 * radius
        posY = (item[1] + 0.5) * 2 * radius
        screen.blit(green_img, (posX, posY))
    for i, bad_item in enumerate(bad_items):
        posX = (bad_item[0] + 0.5) * 2 * radius
        posY = (bad_item[1] + 0.5) * 2 * radius
        screen.blit(red_img, (posX, posY))


def draw_text(screen: pygame.Surface, font: pygame.font.Font, text: str,
              pos: tuple[int], color: tuple[int]) -> None:
    """Render and draw a text string at the given position."""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(topleft=pos)
    screen.blit(text_surface, text_rect)


def add_body_node(snake: list[list], size: int, idx: int) -> None:
    """Add a node to a snake."""
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    rd.shuffle(directions)
    for dx, dy in directions:
        posx = snake[idx][0] + dx
        posy = snake[idx][1] + dy
        if 0 <= posx < size and 0 <= posy < size and [posx, posy] not in snake:
            snake.append([posx, posy])
            return


def create_snake(size: int) -> list[list]:
    """Create the snake, the body has 3 nodes, randomly and continuesly placed
    on the board."""
    head = [rd.randint(0, size - 1), rd.randint(0, size - 1)]
    snake = [head]
    add_body_node(snake, size, 0)
    add_body_node(snake, size, 1)
    return snake


def create_collectible(snake: list[list], size: int) -> list[list]:
    """Initialize 2 collectibles."""
    collectible = []
    add_collectible(collectible, snake, [], size)
    add_collectible(collectible, snake, [], size)
    return collectible


def create_bad_collectible(collectible: list[list],
                           snake: list[list], size: int) -> list[list]:
    """Initialize bad collectible."""
    bad_collectible = []
    add_collectible(bad_collectible, snake, collectible, size)
    return bad_collectible


def add_collectible(collectible: list[list], snake: list[list],
                    other_collect: list[list], size: int) -> None:
    """Add collectible while making sure it doesn't has conflict
    with other collectibles."""
    while True:
        new_col_pos = [rd.randint(0, size - 1), rd.randint(0, size - 1)]
        conflict = False
        if new_col_pos in snake:
            conflict = True
        if new_col_pos in collectible:
            conflict = True
        if new_col_pos in other_collect:
            conflict = True
        if conflict is False:
            collectible.append(new_col_pos)
            break


def init_state(size: int) -> tuple:
    """Initialization of game board, create new snake and collectibles."""
    snake = create_snake(size)
    collectible = create_collectible(snake, size)
    bad_collectible = create_bad_collectible(collectible, snake, size)
    currdir = (0, -1)
    return snake, collectible, bad_collectible, currdir, snake[0]
