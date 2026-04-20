import pygame

class GameBoard:
    def __init__(self, ca, cell_size=10):
        self.ca = ca
        self.width = ca.width
        self.height = ca.height
        self.cell_size = cell_size
        self.running = False
        self.mouse_dragging = False

        self.bg_color = (0, 0, 0)
        self.grid_color = (128, 128, 128)
        self.cell_color = (255, 255, 255)

        self.screen = pygame.display.set_mode((self.width * cell_size,
                                               self.height * cell_size))
        pygame.display.set_caption("Cellular Automaton")

    def draw_grid(self):
        for y in range(0, self.height * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.grid_color,
                             (0, y), (self.width * self.cell_size, y))

        for x in range(0, self.width * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.grid_color,
                             (x, 0), (x, self.height * self.cell_size))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.running = not self.running

        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0]:
            self.mouse_dragging = True
            self.update_cells_on_drag()
        else:
            self.mouse_dragging = False

    def update_cells_on_drag(self):
        if self.mouse_dragging:
            pos = pygame.mouse.get_pos()
            col = pos[0] // self.cell_size
            row = pos[1] // self.cell_size
            self.ca.set_cell_state(row, col, 1)

    def toggle_cell(self, pos):
        x, y = pos
        col = x // self.cell_size
        row = y // self.cell_size
        current_state = self.ca.get_cell_state(row, col)
        new_state = 1 if current_state == 0 else 0
        self.ca.set_cell_state(row, col, new_state)

    def run_simulation(self):
        clock = pygame.time.Clock()
        while True:
            self.handle_events()
            if self.running:
                self.ca.evolve()
            self.screen.fill(self.bg_color)
            self.draw_grid()
            self.draw_cells()
            pygame.display.flip()
            clock.tick(50)

    def draw_cells(self):
        for y in range(self.ca.height):
            for x in range(self.ca.width):
                if self.ca.get_cell_state(y, x):
                    pygame.draw.rect(
                        self.screen,
                        self.cell_color,
                        pygame.Rect(x * self.cell_size,
                                    y * self.cell_size,
                                    self.cell_size,
                                    self.cell_size),
                    )
