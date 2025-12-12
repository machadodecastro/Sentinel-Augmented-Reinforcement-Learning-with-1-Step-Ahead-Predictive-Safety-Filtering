import pygame
import numpy as np
import random

# ------------------ Settings ------------------
CELL_SIZE = 100
GRID_SIZE = 5
SCREEN_SIZE = CELL_SIZE * GRID_SIZE
FPS = 5

# Cores
WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0,255,0)
RED = (255,0,0)
YELLOW = (255,255,0)
BLUE = (0,0,255)

# ------------------ Environment ------------------
class GridEnv:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.state = [0, 0]
        self.rewards = {(4,4):10, (2,2):-10, (3,1):-5}

    def reset(self):
        self.state = [0,0]
        return tuple(self.state)

    def step(self, action):
        x, y = self.state
        if action == 0: x = max(x-1, 0)      # up
        elif action == 1: x = min(x+1, self.size-1)  # down
        elif action == 2: y = max(y-1, 0)    # left
        elif action == 3: y = min(y+1, self.size-1)  # right
        self.state = [x,y]
        reward = self.rewards.get((x,y), 0)
        done = reward != 0 or (x==self.size-1 and y==self.size-1)
        return tuple(self.state), reward, done

# ------------------ Sentinel ------------------
class Sentinel:
    def __init__(self, env):
        self.env = env

    def predict_next(self, state, action):
        x, y = state
        if action == 0: x = max(x-1, 0)
        elif action == 1: x = min(x+1, self.env.size-1)
        elif action == 2: y = max(y-1, 0)
        elif action == 3: y = min(y+1, self.env.size-1)
        reward = self.env.rewards.get((x,y), 0)
        if reward < 0: return -1  # danger
        elif reward > 0: return 1  # positive
        else: return 0  # neutral

# ------------------ Agent ------------------
class Agent:
    def __init__(self, env, sentinel, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.env = env
        self.sentinel = sentinel
        self.q_table = np.zeros((env.size, env.size, 4))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.total_reward = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0,3)
        x,y = state
        return np.argmax(self.q_table[x,y])

    def learn(self, state, action, reward, next_state):
        x,y = state
        nx, ny = next_state
        predict = self.q_table[x,y,action]
        target = reward + self.gamma * np.max(self.q_table[nx,ny])
        self.q_table[x,y,action] += self.alpha * (target - predict)

    def step(self, state):
        # checks for safe actions via sentinel
        safe_actions = []
        for a in range(4):
            signal = self.sentinel.predict_next(state, a)
            if signal != -1:  # avoid danger
                safe_actions.append(a)
        if safe_actions:
            action = random.choice(safe_actions)
        else:
            action = self.choose_action(state)  # there is no safe option
        next_state, reward, done = self.env.step(action)
        self.learn(state, action, reward, next_state)
        self.total_reward += reward
        return next_state, reward, done, action

# ------------------ Pygame Initialization ------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE + 50))
pygame.display.set_caption("RL with Sentinel")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

env = GridEnv()
sentinel = Sentinel(env)
agent = Agent(env, sentinel)

state = env.reset()
done = False

# ------------------ Main Loop ------------------
running = True
while running:
    clock.tick(FPS)
    screen.fill(WHITE)

    # Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Agent performs step
    if not done:
        next_state, reward, done, action = agent.step(state)
    else:
        state = env.reset()
        agent.total_reward = 0
        done = False
        next_state = state

    # Draw grid
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            # pontos positivos e negativos
            if (i,j) in env.rewards:
                color = GREEN if env.rewards[(i,j)]>0 else RED
                pygame.draw.rect(screen, color, rect.inflate(-20,-20))

    # Draw agent
    x,y = next_state
    pygame.draw.circle(screen, BLUE, (y*CELL_SIZE + CELL_SIZE//2, x*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//4)

    # Show sentinel signal for each action
    signals = []
    for a in range(4):
        signal = sentinel.predict_next(next_state, a)
        signals.append(signal)
    # Draw signal arrows
    arrow_coords = [(CELL_SIZE//2,0), (CELL_SIZE//2,CELL_SIZE), (0,CELL_SIZE//2), (CELL_SIZE,CELL_SIZE//2)]
    for idx, s in enumerate(signals):
        color = YELLOW
        if s==1: color=GREEN
        elif s==-1: color=RED
        pygame.draw.line(screen, color,
                         (y*CELL_SIZE + CELL_SIZE//2, x*CELL_SIZE + CELL_SIZE//2),
                         (y*CELL_SIZE + arrow_coords[idx][0], x*CELL_SIZE + arrow_coords[idx][1]), 5)

    # Show reward
    reward_text = font.render(f"Total Reward: {agent.total_reward}", True, BLACK)
    screen.blit(reward_text, (10, SCREEN_SIZE + 10))

    pygame.display.flip()
    state = next_state

pygame.quit()
