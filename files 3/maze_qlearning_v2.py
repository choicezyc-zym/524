"""
Maze Solver using Q-Learning — Advanced Version
CDS524 Assignment 1

Premium dark-tech UI with a large complex maze.
"""

import numpy as np
import pygame
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import time
import math

# ============================================================
#  MAZE  (0=path, 1=wall)  — 20×20 complex maze
# ============================================================
MAZE = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
    [1,0,1,1,1,0,1,0,1,1,1,1,0,1,1,0,1,1,0,1],
    [1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1],
    [1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,0,1,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1],
    [1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1],
    [1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1],
    [1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,1,1,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1],
    [1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,0,1],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1],
    [1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,1],
    [1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,1],
    [1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

START = (1, 1)
GOAL  = (18, 18)
ROWS  = len(MAZE)
COLS  = len(MAZE[0])

ACTIONS      = [0, 1, 2, 3]
ACTION_NAMES = ["↑ Up", "↓ Down", "← Left", "→ Right"]
ACTION_DELTA = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}

# ============================================================
#  Q-LEARNING HYPERPARAMETERS
# ============================================================
ALPHA         = 0.15
GAMMA         = 0.97
EPSILON_START = 1.0
EPSILON_END   = 0.02
EPSILON_DECAY = 0.996
EPISODES      = 1200
MAX_STEPS     = 500

REWARD_GOAL    =  200
REWARD_WALL    =  -15
REWARD_STEP    =   -1
REWARD_REVISIT =   -3

# ============================================================
#  PYGAME / UI CONFIG
# ============================================================
CELL      = 36          # Cell pixel size
PANEL_W   = 340         # Right info panel
BOTTOM_H  = 160         # Bottom stats bar
MAZE_W    = COLS * CELL
MAZE_H    = ROWS * CELL
WIN_W     = MAZE_W + PANEL_W
WIN_H     = MAZE_H + BOTTOM_H

# Dark-tech color palette
BG         = (8,   12,  22)
WALL       = (20,  28,  48)
WALL_EDGE  = (35,  55,  90)
PATH       = (14,  20,  36)
PATH_GRID  = (22,  32,  56)
START_C    = (30, 200, 120)
GOAL_C     = (255, 80,  80)
AGENT_C    = (0,  180, 255)
AGENT_GLOW = (0,   80, 180)
TRAIL_C    = (0,   60, 120)
TRAIL_HEAD = (0,  130, 220)
PANEL_BG   = (10,  15,  28)
PANEL_SEP  = (30,  45,  75)
TEXT_PRI   = (210, 225, 255)
TEXT_SEC   = (100, 130, 180)
TEXT_ACC   = (0,  200, 255)
TEXT_OK    = (50,  220, 130)
TEXT_WARN  = (255, 140,  50)
TEXT_ERR   = (255,  70,  70)
CHART_BG   = (12,  18,  34)
CHART_LINE = (0,  180, 255)
CHART_FILL = (0,   50, 100)


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i]-c1[i])*t) for i in range(3))


# ============================================================
#  ENVIRONMENT
# ============================================================
class MazeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos     = START
        self.visited = {START}
        return self.pos

    def step(self, action):
        dr, dc = ACTION_DELTA[action]
        nr, nc = self.pos[0]+dr, self.pos[1]+dc
        if nr<0 or nr>=ROWS or nc<0 or nc>=COLS or MAZE[nr][nc]==1:
            return self.pos, REWARD_WALL, False
        self.pos = (nr, nc)
        if self.pos == GOAL:
            return self.pos, REWARD_GOAL, True
        reward = REWARD_REVISIT if self.pos in self.visited else REWARD_STEP
        self.visited.add(self.pos)
        return self.pos, reward, False

    def valid_states(self):
        return [(r,c) for r in range(ROWS) for c in range(COLS) if MAZE[r][c]==0]


# ============================================================
#  Q-LEARNING AGENT
# ============================================================
class QLearningAgent:
    def __init__(self):
        env = MazeEnv()
        self.q  = {s: np.zeros(4) for s in env.valid_states()}
        self.eps = EPSILON_START

    def act(self, s):
        if random.random() < self.eps:
            return random.randint(0,3)
        return int(np.argmax(self.q[s]))

    def update(self, s, a, r, ns, done):
        tgt = r if done else r + GAMMA * np.max(self.q[ns])
        self.q[s][a] += ALPHA * (tgt - self.q[s][a])

    def decay(self):
        self.eps = max(EPSILON_END, self.eps * EPSILON_DECAY)

    def greedy_path(self):
        env = MazeEnv(); s = env.reset()
        path = [s]; seen = {s}
        for _ in range(MAX_STEPS):
            a = int(np.argmax(self.q[s]))
            ns, _, done = env.step(a)
            if ns in seen: break
            path.append(ns); seen.add(ns); s = ns
            if done: break
        return path


# ============================================================
#  TRAINING
# ============================================================
def train(agent):
    env = MazeEnv()
    rews, stps = [], []
    wins = 0
    for ep in range(EPISODES):
        s = env.reset(); tot = 0; st = 0
        for _ in range(MAX_STEPS):
            a = agent.act(s)
            ns, r, done = env.step(a)
            agent.update(s, a, r, ns, done)
            s = ns; tot += r; st += 1
            if done: wins += 1; break
        agent.decay()
        rews.append(tot); stps.append(st)
    return rews, stps, wins


# ============================================================
#  RENDERER
# ============================================================
class Renderer:
    def __init__(self, agent, rewards, steps_history):
        pygame.init()
        self.win     = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Q-Learning Maze Agent  |  CDS524")
        self.agent   = agent
        self.rewards = rewards
        self.steps_h = steps_history
        self.tick    = 0

        # Fonts
        self.f_title  = pygame.font.SysFont("Courier New", 18, bold=True)
        self.f_label  = pygame.font.SysFont("Courier New", 13)
        self.f_small  = pygame.font.SysFont("Courier New", 11)
        self.f_big    = pygame.font.SysFont("Courier New", 22, bold=True)
        self.f_huge   = pygame.font.SysFont("Courier New", 28, bold=True)

        # Pre-render maze surface
        self.maze_surf = self._build_maze_surf()

        # Chart surface cache
        self._chart_surf = None
        self._build_chart()

    def _build_maze_surf(self):
        surf = pygame.Surface((MAZE_W, MAZE_H))
        surf.fill(BG)
        for r in range(ROWS):
            for c in range(COLS):
                rect = pygame.Rect(c*CELL, r*CELL, CELL, CELL)
                if MAZE[r][c] == 1:
                    pygame.draw.rect(surf, WALL, rect)
                    pygame.draw.rect(surf, WALL_EDGE, rect, 1)
                else:
                    pygame.draw.rect(surf, PATH, rect)
                    pygame.draw.rect(surf, PATH_GRID, rect, 1)
        return surf

    def _build_chart(self):
        """Draw reward curve as a pygame surface."""
        W, H = PANEL_W - 20, 90
        surf = pygame.Surface((W, H))
        surf.fill(CHART_BG)
        pygame.draw.rect(surf, PANEL_SEP, (0,0,W,H), 1)

        if len(self.rewards) < 5:
            self._chart_surf = surf
            return

        win = min(40, len(self.rewards))
        smoothed = np.convolve(self.rewards, np.ones(win)/win, mode='valid')
        mn, mx = min(smoothed), max(smoothed)
        if mx == mn: mx = mn + 1

        pts = []
        for i, v in enumerate(smoothed):
            x = int(i / len(smoothed) * (W-4)) + 2
            y = int((1 - (v-mn)/(mx-mn)) * (H-8)) + 4
            pts.append((x, y))

        if len(pts) > 1:
            # Fill area under curve
            poly = [(pts[0][0], H-2)] + pts + [(pts[-1][0], H-2)]
            fill_surf = pygame.Surface((W, H), pygame.SRCALPHA)
            pygame.draw.polygon(fill_surf, (*CHART_FILL, 120), poly)
            surf.blit(fill_surf, (0,0))
            pygame.draw.lines(surf, CHART_LINE, False, pts, 2)

        # Labels
        mn_lbl = self.f_small.render(f"{mn:.0f}", True, TEXT_SEC)
        mx_lbl = self.f_small.render(f"{mx:.0f}", True, TEXT_SEC)
        surf.blit(mx_lbl, (3, 2))
        surf.blit(mn_lbl, (3, H-14))

        self._chart_surf = surf

    def _glow_circle(self, surf, color, glow_color, center, radius):
        """Draw a circle with a subtle glow."""
        for r in range(radius+8, radius-1, -2):
            alpha = int(60 * (1 - (r-radius)/8)) if r > radius else 255
            t     = max(0, min(1, (r-radius)/8))
            c     = lerp_color(color, glow_color, t) if r > radius else color
            pygame.draw.circle(surf, c, center, r)

    def draw(self, pos, trail, reward, steps, action, episode, done, success_count):
        self.tick += 1
        win = self.win

        # ── Background ──
        win.fill(BG)

        # ── Maze tiles ──
        win.blit(self.maze_surf, (0, 0))

        # ── Trail ──
        for i, p in enumerate(trail):
            t   = i / max(len(trail), 1)
            col = lerp_color(TRAIL_C, TRAIL_HEAD, t)
            cx  = p[1]*CELL + CELL//2
            cy  = p[0]*CELL + CELL//2
            r   = max(3, int(CELL//2 * 0.45 * (0.4 + 0.6*t)))
            pygame.draw.circle(win, col, (cx, cy), r)

        # ── Start marker ──
        sx = START[1]*CELL + CELL//2
        sy = START[0]*CELL + CELL//2
        pygame.draw.circle(win, START_C, (sx, sy), CELL//2 - 4)
        lbl = self.f_label.render("S", True, (10,10,10))
        win.blit(lbl, (sx - lbl.get_width()//2, sy - lbl.get_height()//2))

        # ── Goal marker ──
        gx = GOAL[1]*CELL + CELL//2
        gy = GOAL[0]*CELL + CELL//2
        pulse = abs(math.sin(self.tick * 0.07))
        gr_r  = int(CELL//2 - 4 + pulse * 4)
        pygame.draw.circle(win, GOAL_C, (gx, gy), gr_r)
        lbl = self.f_label.render("G", True, (255,255,255))
        win.blit(lbl, (gx - lbl.get_width()//2, gy - lbl.get_height()//2))

        # ── Agent (animated) ──
        ax = pos[1]*CELL + CELL//2
        ay = pos[0]*CELL + CELL//2
        agent_r = CELL//2 - 5
        self._glow_circle(win, AGENT_C, AGENT_GLOW, (ax, ay), agent_r)
        # Inner white dot
        pygame.draw.circle(win, (200, 240, 255), (ax, ay), 4)

        # ── RIGHT PANEL ──
        px = MAZE_W
        pygame.draw.rect(win, PANEL_BG, (px, 0, PANEL_W, MAZE_H))
        pygame.draw.line(win, PANEL_SEP, (px, 0), (px, MAZE_H), 2)

        def txt(text, y, color=TEXT_PRI, font=None, x_off=18):
            f = font or self.f_label
            s = f.render(text, True, color)
            win.blit(s, (px + x_off, y))

        def sep_line(y):
            pygame.draw.line(win, PANEL_SEP, (px+10, y), (px+PANEL_W-10, y), 1)

        # Title
        txt("[ Q-LEARNING AGENT ]", 16, TEXT_ACC, self.f_title)
        txt("CDS524 Assignment 1", 38, TEXT_SEC, self.f_small)
        sep_line(56)

        # Episode stats
        txt("EPISODE STATS", 66, TEXT_SEC, self.f_small)
        txt(f"Episode    {episode:>6}", 84, TEXT_PRI)
        txt(f"Steps      {steps:>6}", 102, TEXT_PRI)
        rew_col = TEXT_OK if reward >= 0 else TEXT_ERR
        txt(f"Reward  {reward:>+9.0f}", 120, rew_col)
        txt(f"Action     {action:>6}", 138, TEXT_PRI)
        sep_line(158)

        # Agent metrics
        txt("AGENT METRICS", 166, TEXT_SEC, self.f_small)
        txt(f"Epsilon    {self.agent.eps:>6.4f}", 184)
        txt(f"Successes  {success_count:>6}", 202, TEXT_OK)
        txt(f"Position  {str(pos):>7}", 220)
        sep_line(240)

        # Hyperparams
        txt("HYPERPARAMETERS", 248, TEXT_SEC, self.f_small)
        txt(f"α  (learn rate)   {ALPHA}", 266)
        txt(f"γ  (discount)     {GAMMA}", 284)
        txt(f"ε  {EPSILON_START} → {EPSILON_END}", 302)
        txt(f"Episodes       {EPISODES}", 320)
        sep_line(340)

        # Reward chart
        txt("REWARD CURVE", 348, TEXT_SEC, self.f_small)
        if self._chart_surf:
            win.blit(self._chart_surf, (px+10, 364))
        sep_line(462)

        # Status
        txt("STATUS", 470, TEXT_SEC, self.f_small)
        if done and pos == GOAL:
            txt("●  GOAL REACHED!", 488, TEXT_OK, self.f_title)
        elif done:
            txt("●  EPISODE ENDED", 488, TEXT_WARN)
        else:
            blink = (self.tick // 15) % 2
            dot   = "●" if blink else "○"
            txt(f"{dot}  NAVIGATING...", 488, AGENT_C)

        # Legend
        sep_line(510)
        txt("LEGEND", 518, TEXT_SEC, self.f_small)
        pygame.draw.circle(win, START_C, (px+24, 536), 7)
        txt("Start point", 530, TEXT_PRI, x_off=36)
        pygame.draw.circle(win, GOAL_C,  (px+24, 556), 7)
        txt("Goal point",  550, TEXT_PRI, x_off=36)
        pygame.draw.circle(win, AGENT_C, (px+24, 576), 7)
        txt("Agent",       570, TEXT_PRI, x_off=36)
        pygame.draw.circle(win, TRAIL_HEAD, (px+24, 596), 5)
        txt("Trail",       590, TEXT_PRI, x_off=36)

        # ── BOTTOM STATS BAR ──
        by = MAZE_H
        pygame.draw.rect(win, PANEL_BG, (0, by, WIN_W, BOTTOM_H))
        pygame.draw.line(win, PANEL_SEP, (0, by), (WIN_W, by), 2)

        # Stats blocks
        stats = [
            ("MAZE SIZE",     f"{ROWS}×{COLS}"),
            ("STATE SPACE",   f"{ROWS*COLS} cells"),
            ("ACTION SPACE",  "4 directions"),
            ("TOTAL EPISODES",f"{EPISODES}"),
            ("PATH LENGTH",   f"{steps} steps"),
            ("TOTAL REWARD",  f"{reward:+.0f}"),
        ]
        block_w = (MAZE_W) // len(stats)
        for i, (label, val) in enumerate(stats):
            bx = i * block_w
            if i > 0:
                pygame.draw.line(win, PANEL_SEP, (bx, by+10), (bx, by+BOTTOM_H-10), 1)
            lbl_s = self.f_small.render(label, True, TEXT_SEC)
            val_s = self.f_big.render(val,   True, TEXT_ACC)
            win.blit(lbl_s, (bx + (block_w - lbl_s.get_width())//2, by + 22))
            win.blit(val_s, (bx + (block_w - val_s.get_width())//2,  by + 42))

        # Bottom right: Q-table info
        bx2 = MAZE_W + 10
        txt2 = self.f_small.render("Q-TABLE", True, TEXT_SEC)
        win.blit(txt2, (bx2, by + 16))
        n_nonzero = sum(1 for v in self.agent.q.values() if np.any(v != 0))
        v1 = self.f_label.render(f"Entries: {len(self.agent.q)}", True, TEXT_PRI)
        v2 = self.f_label.render(f"Non-zero: {n_nonzero}", True, TEXT_OK)
        v3 = self.f_label.render(f"Max Q: {max(np.max(v) for v in self.agent.q.values()):.1f}", True, TEXT_ACC)
        win.blit(v1, (bx2, by + 36))
        win.blit(v2, (bx2, by + 56))
        win.blit(v3, (bx2, by + 76))

        pygame.display.flip()

    def run_episode(self, path, episode, success_count, reward_total):
        """Replay a path step by step."""
        trail  = []
        steps  = 0
        done   = False
        clock  = pygame.time.Clock()

        for idx, pos in enumerate(path):
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return False
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    pass  # future: pause

            if idx > 0:
                prev  = path[idx-1]
                dr    = pos[0] - prev[0]
                dc    = pos[1] - prev[1]
                for a, (ddr, ddc) in ACTION_DELTA.items():
                    if ddr == dr and ddc == dc:
                        act_name = ACTION_NAMES[a]; break
                else:
                    act_name = "—"
            else:
                act_name = "—"

            trail.append(pos)
            steps = idx + 1
            done  = (pos == GOAL)

            self.draw(pos, trail[:-1], reward_total, steps,
                      act_name, episode, done, success_count)
            clock.tick(12)  # ~12fps for smooth animation

        pygame.time.delay(600)
        return True


# ============================================================
#  LOADING SCREEN
# ============================================================
def show_loading(win, fonts, episode, total, eps, avg_r):
    f_title = fonts[0]; f_lbl = fonts[1]; f_small = fonts[2]
    win.fill(BG)

    # Title
    t = f_title.render("TRAINING Q-LEARNING AGENT", True, TEXT_ACC)
    win.blit(t, ((WIN_W - t.get_width())//2, 80))

    sub = f_lbl.render("CDS524 Assignment 1  —  Maze Solver", True, TEXT_SEC)
    win.blit(sub, ((WIN_W - sub.get_width())//2, 115))

    # Progress bar
    bar_w = 500; bar_h = 18
    bx    = (WIN_W - bar_w)//2; by = 200
    prog  = episode / total
    pygame.draw.rect(win, PANEL_SEP, (bx, by, bar_w, bar_h), border_radius=9)
    if prog > 0:
        pygame.draw.rect(win, AGENT_C, (bx, by, int(bar_w*prog), bar_h), border_radius=9)
    pygame.draw.rect(win, PANEL_SEP, (bx, by, bar_w, bar_h), 2, border_radius=9)

    p_lbl = f_lbl.render(f"{episode}/{total}  ({prog*100:.1f}%)", True, TEXT_PRI)
    win.blit(p_lbl, ((WIN_W - p_lbl.get_width())//2, by + 26))

    # Stats
    info = [
        ("EPSILON",      f"{eps:.4f}"),
        ("AVG REWARD",   f"{avg_r:.1f}"),
        ("EPISODES",     f"{total}"),
        ("MAZE",         f"{ROWS}×{COLS}"),
    ]
    iw = 200
    ix = (WIN_W - iw*len(info))//2
    for i, (lbl, val) in enumerate(info):
        x = ix + i*iw
        l = f_small.render(lbl, True, TEXT_SEC)
        v = fonts[3].render(val, True, TEXT_ACC)
        win.blit(l, (x + (iw-l.get_width())//2, 290))
        win.blit(v, (x + (iw-v.get_width())//2, 310))

    hint = f_small.render("Training in progress... please wait", True, TEXT_SEC)
    win.blit(hint, ((WIN_W - hint.get_width())//2, WIN_H - 60))

    pygame.display.flip()


# ============================================================
#  MAIN
# ============================================================
def main():
    pygame.init()
    win   = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Q-Learning Maze  |  CDS524  |  Training...")

    f_title = pygame.font.SysFont("Courier New", 22, bold=True)
    f_lbl   = pygame.font.SysFont("Courier New", 16)
    f_small = pygame.font.SysFont("Courier New", 12)
    f_big   = pygame.font.SysFont("Courier New", 26, bold=True)
    fonts   = [f_title, f_lbl, f_small, f_big]

    agent   = QLearningAgent()
    env     = MazeEnv()
    rewards = []; steps_h = []; wins = 0
    t0      = time.time()

    print("=" * 55)
    print("   CDS524 Assignment 1 — Maze Q-Learning (Advanced)")
    print("=" * 55)
    print(f"   Maze       : {ROWS}×{COLS}")
    print(f"   Start      : {START}  →  Goal : {GOAL}")
    print(f"   Episodes   : {EPISODES}")
    print(f"   α={ALPHA}  γ={GAMMA}  ε:{EPSILON_START}→{EPSILON_END}")
    print()

    # ── Train with live agent visualization ──
    # Show every step for first 50 eps, then every 10th ep, then every 50th
    renderer_live = Renderer(agent, rewards, steps_h)
    clock_train   = pygame.time.Clock()

    for ep in range(EPISODES):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        s = env.reset(); tot = 0; st = 0
        trail = []

        # Decide render frequency based on training phase
        if ep < 50:
            render_every = 1      # Show every step — agent is learning basics
        elif ep < 300:
            render_every = 3      # Show every 3 steps
        else:
            render_every = 999    # Fast train, only show episode end

        for step in range(MAX_STEPS):
            a  = agent.act(s)
            ns, r, done = env.step(a)
            agent.update(s, a, r, ns, done)
            trail.append(s)
            s = ns; tot += r; st += 1

            # Live render during early training
            if step % render_every == 0 or done:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit(); sys.exit()
                act_name = ACTION_NAMES[a]
                renderer_live.draw(s, trail[-30:], tot, st,
                                   act_name, ep+1, done, wins)
                if ep < 50:
                    clock_train.tick(30)   # 30fps during slow phase
                elif ep < 300:
                    clock_train.tick(60)

            if done:
                wins += 1
                break

        agent.decay()
        rewards.append(tot); steps_h.append(st)
        renderer_live.rewards    = rewards
        renderer_live.steps_h    = steps_h
        renderer_live._build_chart()

        if (ep+1) % 200 == 0:
            avg_r = np.mean(rewards[-50:])
            print(f"   Ep {ep+1:4d}/{EPISODES}  ε={agent.eps:.3f}  "
                  f"AvgR={avg_r:7.1f}  Wins={wins}")

    elapsed = time.time() - t0
    print(f"\n   Training done in {elapsed:.1f}s")
    print(f"   Goal reached: {wins}/{EPISODES} episodes")

    # ── Save chart ──
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7))
    fig.patch.set_facecolor('#080c16')
    for ax in (a1, a2):
        ax.set_facecolor('#0c1222')
        ax.tick_params(colors='#6482b4')
        for sp in ax.spines.values():
            sp.set_edgecolor('#1e2d4a')

    w = 40
    sr = np.convolve(rewards,   np.ones(w)/w, mode='valid')
    ss = np.convolve(steps_h,   np.ones(w)/w, mode='valid')
    xs = range(w-1, len(rewards))

    a1.plot(rewards, alpha=0.15, color='#00b4ff')
    a1.plot(xs, sr, color='#00b4ff', lw=2)
    a1.fill_between(xs, sr, alpha=0.15, color='#00b4ff')
    a1.set_title('Episode Reward', color='#00c8ff', fontsize=12)
    a1.set_ylabel('Reward', color='#6482b4')

    a2.plot(steps_h, alpha=0.15, color='#50dc82')
    a2.plot(xs, ss, color='#50dc82', lw=2)
    a2.fill_between(xs, ss, alpha=0.15, color='#50dc82')
    a2.set_title('Steps per Episode', color='#00c8ff', fontsize=12)
    a2.set_ylabel('Steps', color='#6482b4')
    a2.set_xlabel('Episode', color='#6482b4')

    plt.tight_layout(pad=2)
    plt.savefig("training_results.png", dpi=150, facecolor='#080c16')
    plt.close()
    print("   Chart saved: training_results.png")

    # ── Visualization loop ──
    pygame.display.set_caption("Q-Learning Maze  |  CDS524")
    renderer = Renderer(agent, rewards, steps_h)
    best     = agent.greedy_path()
    print(f"\n   Best path length: {len(best)} steps")
    print(f"   Goal reached: {best[-1] == GOAL}")
    print("\n   Launching visualization... (close window to quit)\n")

    ep_num  = 1
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        r_total = sum(REWARD_STEP for _ in best[:-1])
        if best[-1] == GOAL:
            r_total += REWARD_GOAL

        ok = renderer.run_episode(best, ep_num, wins, r_total)
        if not ok:
            running = False
        ep_num  = ep_num % 99 + 1

    pygame.quit()
    print("Done!")


if __name__ == "__main__":
    main()
