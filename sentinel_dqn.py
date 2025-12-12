import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------ Settings ------------------
CELL_SIZE = 100
GRID_SIZE = 5
SCREEN_SIZE = CELL_SIZE * GRID_SIZE
FPS = 6  # speed

# colors
WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0,200,0)
RED = (200,30,30)
YELLOW = (220,180,20)
BLUE = (0,100,255)
GHOST_SENT = (120,180,255,160)  # sentinel color (alpha)

# ------------------ Environment ------------------
class GridEnv:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.state = [0, 0]
        # original rewards
        self.rewards = {(4,4):10, (2,2):-10, (3,1):-5}

    def reset(self):
        self.state = [0,0]
        return tuple(self.state)

    def step(self, action):
        x, y = self.state
        if action == 0: x = max(x-1, 0)      # up
        elif action == 1: x = min(x+1, self.size-1)  # down
        elif action == 2: y = max(y-1, 0)    # left
        elif action == 3: y = min(y+1, self.size-1)  # dirightreita
        self.state = [x,y]
        reward = self.rewards.get((x,y), 0)
        done = reward != 0 or (x==self.size-1 and y==self.size-1)
        return tuple(self.state), float(reward), done

# ------------------ Sentinel (keeps one step ahead) ------------------
class Sentinel:
    def __init__(self, env):
        self.env = env

    def predict_next(self, state, action):
        # It simulates the next position and returns a signal
        x, y = state
        if action == 0: x = max(x-1, 0)
        elif action == 1: x = min(x+1, self.env.size-1)
        elif action == 2: y = max(y-1, 0)
        elif action == 3: y = min(y+1, self.env.size-1)
        reward = self.env.rewards.get((x,y), 0)
        if reward < 0:
            return -1  # danger
        elif reward > 0:
            return 1   # positive
        else:
            return 0   # neutral

    def next_pos(self, state, action):
        # returns the simulated position (x,y) to draw the "sentinel ghost"
        x, y = state
        if action == 0: x = max(x-1, 0)
        elif action == 1: x = min(x+1, self.env.size-1)
        elif action == 2: y = max(y-1, 0)
        elif action == 3: y = min(y+1, self.env.size-1)
        return (x, y)

# ------------------ Simple DQN (MLP) ------------------
class SimpleDQN(nn.Module):
    def __init__(self, input_dim, hidden=64, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ------------------ Agent DQN (online update, no replay, no target) ------------------
class DQNAgentSimple:
    def __init__(self, env, sentinel, lr=1e-3, gamma=0.99, epsilon=0.5, eps_min=0.02, eps_decay=0.995, device='cpu'):
        self.env = env
        self.sentinel = sentinel
        self.device = torch.device(device)
        # input: one-hot pos (25) + sentinel signals (4) => 29
        self.state_dim = env.size * env.size + 4
        self.action_dim = 4

        self.model = SimpleDQN(self.state_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.total_reward = 0.0

    def state_to_vec(self, state):
        # one-hot position
        vec = np.zeros(self.env.size * self.env.size, dtype=np.float32)
        r,c = state
        vec[r * self.env.size + c] = 1.0
        # sentinel signals for each action (values -1,0,1)
        sigs = np.array([self.sentinel.predict_next(state, a) for a in range(4)], dtype=np.float32)
        # concat
        full = np.concatenate([vec, sigs], axis=0)
        return full  # shape = 29

    def select_action(self, state_vec, safe_actions):
        # safe_actions: list of allowed action indices (filtered by sentinel)
        # epsilon-greedy among safe_actions if non-empty, else among all actions
        if len(safe_actions) == 0:
            candidates = list(range(self.action_dim))
        else:
            candidates = safe_actions

        if random.random() < self.epsilon:
            return random.choice(candidates)
        # forward
        self.model.eval()
        with torch.no_grad():
            s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(s).cpu().numpy().flatten()
        # choose best among candidates
        best = max(candidates, key=lambda a: q[a])
        return int(best)

    def train_step(self, state_vec, action, reward, next_state_vec, done):
        # compute target: r + gamma * max_a' Q(next)
        self.model.train()
        s = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_vals = self.model(s)  # shape [1,4]

        with torch.no_grad():
            s2 = torch.tensor(next_state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_next = self.model(s2)  # no target network in simple DQN
            max_q_next = torch.max(q_next).unsqueeze(0)

            target_val = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
            if not done:
                target_val = target_val + (self.gamma * max_q_next)

        # create expected full vector with only the taken action updated
        target_full = q_vals.clone().detach()
        target_full[0, action] = target_val

        loss = self.loss_fn(q_vals, target_full)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epsilon
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

    def save(self, path):
        torch.save({'model': self.model.state_dict(), 'epsilon': self.epsilon}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.epsilon = data.get('epsilon', self.epsilon)

# ------------------ Pygame Initialization ------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE + 50))
pygame.display.set_caption("DQN Simples + Sentinela (mantendo jogo original)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)

env = GridEnv()
sentinel = Sentinel(env)
agent = DQNAgentSimple(env, sentinel, lr=1e-3, gamma=0.95, epsilon=0.6, eps_decay=0.995)

state = env.reset()
done = False
episode = 0
episode_reward = 0.0

running = True
while running:
    clock.tick(FPS)
    screen.fill(WHITE)

    # events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # construct current state vector (with sentinel signals)
    state_vec = agent.state_to_vec(state)

    # the sentinel tests all actions and defines safe actions
    signals = [sentinel.predict_next(state, a) for a in range(4)]
    safe_actions = [a for a, s in enumerate(signals) if s != -1]  # evita ações sinalizadas como perigo

    # select action (filtering hazards)
    action = agent.select_action(state_vec, safe_actions)

    # draw a preview of the "sentinel ghosts" in each tested action
    preview_positions = [sentinel.next_pos(state, a) for a in range(4)]

    # step into the environment with the chosen action
    next_state, reward, done = env.step(action)
    episode_reward += reward
    agent.total_reward += reward

    # next vector state
    next_state_vec = agent.state_to_vec(next_state)

    # train online (one step)
    agent.train_step(state_vec, action, reward, next_state_vec, done)

    # DRAWING: grid and rewards
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            if (i,j) in env.rewards:
                color = GREEN if env.rewards[(i,j)]>0 else RED
                pygame.draw.rect(screen, color, rect.inflate(-20,-20))

    # draw agent
    ar, ac = next_state
    pygame.draw.circle(screen, BLUE, (ac*CELL_SIZE + CELL_SIZE//2, ar*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//4)

    # Draw sentinel previews (a small ghost for each action) and arrows
    arrow_coords = [(CELL_SIZE//2,0), (CELL_SIZE//2,CELL_SIZE), (0,CELL_SIZE//2), (CELL_SIZE,CELL_SIZE//2)]
    for idx, (pr,pc) in enumerate(preview_positions):
        # color by signal
        s = signals[idx]
        if s == 1:
            col = GREEN
        elif s == -1:
            col = RED
        else:
            col = YELLOW
        # draw a small "ghost" (rounded rectangle)
        center = (pc*CELL_SIZE + CELL_SIZE//2, pr*CELL_SIZE + CELL_SIZE//2)
        # split circle for visual ghost effect
        ghost_radius = CELL_SIZE//6
        ghost_surf = pygame.Surface((ghost_radius*2, ghost_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(ghost_surf, (col[0], col[1], col[2], 160), (ghost_radius, ghost_radius), ghost_radius)
        screen.blit(ghost_surf, (center[0]-ghost_radius, center[1]-ghost_radius))
        # agent arrow (from current agent to predicted position)
        # draw arrows from the agent's current position (state)
        cr, cc = state
        start_pos = (cc*CELL_SIZE + CELL_SIZE//2, cr*CELL_SIZE + CELL_SIZE//2)
        end_pos = (pc*CELL_SIZE + CELL_SIZE//2, pr*CELL_SIZE + CELL_SIZE//2)
        pygame.draw.line(screen, col, start_pos, end_pos, 4)

    # Reward
    info = f"Epi: {episode}  EpReward: {episode_reward:.1f}  TotalReward: {agent.total_reward:.1f}  Eps: {agent.epsilon:.3f}"
    txt = font.render(info, True, BLACK)
    screen.blit(txt, (8, SCREEN_SIZE + 12))

    pygame.display.flip()

    # update state
    state = next_state

    if done:
        episode += 1
        # simple reset
        state = env.reset()
        episode_reward = 0.0
        done = False
        # optional: save template a few times
        if episode % 200 == 0:
            try:
                agent.save("dqn_simple_sent.pth")
                print("Saved model in dqn_simple_sent.pth (ep {})".format(episode))
            except Exception as e:
                print("Error saving:", e)

pygame.quit()
