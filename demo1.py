import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
from collections import deque

# --- 1. DOMAIN CONFIGURATION ---
WIDTH, HEIGHT = 200, 100
STEPS = 4000

# FHP D2Q6 hexagonal lattice
#   2     1
#     \  /
#  3 -- X -- 0
#     /  \
#   4     5
#
# Channel 0: East  (E)
# Channel 1: Northeast (NE)
# Channel 2: Northwest (NW)
# Channel 3: West  (W)
# Channel 4: Southwest (SW)
# Channel 5: Southeast (SE)
grid = np.zeros((6, HEIGHT, WIDTH), dtype=bool)

# Barrier at x = 100 with a gap in the middle
barrier_x = WIDTH // 2
barrier_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
barrier_mask[:, barrier_x] = True

gap_half_size = 15
gap_start = HEIGHT // 2 - gap_half_size
gap_end   = HEIGHT // 2 + gap_half_size
barrier_mask[gap_start:gap_end, barrier_x] = False

# Barrier face masks for bounce-back
left_face  = np.zeros((HEIGHT, WIDTH), dtype=bool)
right_face = np.zeros((HEIGHT, WIDTH), dtype=bool)
left_face[:, :-1]  = ~barrier_mask[:, :-1] &  barrier_mask[:, 1:]
right_face[:, 1:] = ~barrier_mask[:, 1:]  &  barrier_mask[:, :-1]



# --- 2. INITIALISATION ---
probability_of_particle = 0.4

def init_grid():
    g = np.zeros((6, HEIGHT, WIDTH), dtype=bool)
    for c in range(6):
        g[c, :, :barrier_x] = np.random.rand(HEIGHT, barrier_x) < probability_of_particle
    return g

grid = init_grid()

# History buffer for reverse playback
MAX_HISTORY = 4000
history = deque(maxlen=MAX_HISTORY)

# --- 3. FHP ENGINE ---
def step_fhp(current_grid, barrier, lface, rface):
    n = current_grid.copy()
    H, W = n.shape[1], n.shape[2]
    even = np.arange(H) % 2 == 0
    odd = ~even
    even_rows = np.where(even)[0]
    odd_rows = np.where(odd)[0]

    # --- COLLISION (FHP-I) ---
    triple_024 = n[0] & n[2] & n[4] & ~(n[1] | n[3] | n[5])
    triple_135 = n[1] & n[3] & n[5] & ~(n[0] | n[2] | n[4])

    headon_03 = n[0] & n[3] & ~(n[1] | n[2] | n[4] | n[5])
    headon_14 = n[1] & n[4] & ~(n[0] | n[2] | n[3] | n[5])
    headon_25 = n[2] & n[5] & ~(n[0] | n[1] | n[3] | n[4])

    # Triple: d0,d2,d4 <-> d1,d3,d5
    for c in (0, 2, 4):
        n[c] = np.where(triple_024, False, n[c])
    for c in (1, 3, 5):
        n[c] = np.where(triple_024, True, n[c])
    for c in (1, 3, 5):
        n[c] = np.where(triple_135, False, n[c])
    for c in (0, 2, 4):
        n[c] = np.where(triple_135, True, n[c])

    # Head-on d0+d3 -> 50% (d1,d4), 50% (d2,d5)
    rand_03 = np.random.rand(H, W) < 0.5
    n[0] = np.where(headon_03, False, n[0])
    n[3] = np.where(headon_03, False, n[3])
    n[1] = np.where(headon_03 & rand_03, True, n[1])
    n[4] = np.where(headon_03 & rand_03, True, n[4])
    n[2] = np.where(headon_03 & ~rand_03, True, n[2])
    n[5] = np.where(headon_03 & ~rand_03, True, n[5])

    # Head-on d1+d4 -> 50% (d2,d5), 50% (d0,d3)
    rand_14 = np.random.rand(H, W) < 0.5
    n[1] = np.where(headon_14, False, n[1])
    n[4] = np.where(headon_14, False, n[4])
    n[2] = np.where(headon_14 & rand_14, True, n[2])
    n[5] = np.where(headon_14 & rand_14, True, n[5])
    n[0] = np.where(headon_14 & ~rand_14, True, n[0])
    n[3] = np.where(headon_14 & ~rand_14, True, n[3])

    # Head-on d2+d5 -> 50% (d0,d3), 50% (d1,d4)
    rand_25 = np.random.rand(H, W) < 0.5
    n[2] = np.where(headon_25, False, n[2])
    n[5] = np.where(headon_25, False, n[5])
    n[0] = np.where(headon_25 & rand_25, True, n[0])
    n[3] = np.where(headon_25 & rand_25, True, n[3])
    n[1] = np.where(headon_25 & ~rand_25, True, n[1])
    n[4] = np.where(headon_25 & ~rand_25, True, n[4])

    # --- BARRIER FACE REFLECTION (bounce-back) ---
    # Left face: E(d0), NE(d1), SE(d5) -> W(d3), NW(d2), SW(d4)
    ref = lface & n[0]; n[0] &= ~ref; n[3] |= ref
    ref = lface & n[1]; n[1] &= ~ref; n[2] |= ref
    ref = lface & n[5]; n[5] &= ~ref; n[4] |= ref

    # Right face: W(d3), NW(d2), SW(d4) -> E(d0), NE(d1), SE(d5)
    ref = rface & n[3]; n[3] &= ~ref; n[0] |= ref
    ref = rface & n[2]; n[2] &= ~ref; n[1] |= ref
    ref = rface & n[4]; n[4] &= ~ref; n[5] |= ref

    n[:, barrier] = False

    # --- STREAMING WITH OUTER WALL BOUNCE-BACK ---
    # The hexagonal lattice uses parity-dependent neighbor offsets:
    # Even rows: E=(+1,0) NE=(0,-1) NW=(-1,-1) W=(-1,0) SW=(-1,+1) SE=(0,+1)
    # Odd rows:  E=(+1,0) NE=(+1,-1) NW=(0,-1) W=(-1,0) SW=(0,+1) SE=(+1,+1)
    next_grid = np.zeros_like(current_grid)

    # d0 (E): parity independent, stream right; right wall -> d3
    next_grid[0, :, 1:] = n[0, :, :-1]
    next_grid[3, :, -1] |= n[0, :, -1]

    # d3 (W): parity independent, stream left; left wall -> d0
    next_grid[3, :, :-1] = n[3, :, 1:]
    next_grid[0, :, 0] |= n[3, :, 0]

    # d1 (NE): even -> (y-1,x), odd -> (y-1,x+1)
    n1_even = n[1, even, :]
    # Even rows y>0 -> odd targets (y-1)
    if len(even_rows) > 1:
        next_grid[1, odd_rows[:-1], :] = n1_even[1:, :]
    # y=0 (top) bounce to d5
    next_grid[4, 0, :] |= n1_even[0, :]
    # Odd rows: bounce right wall (x=W-1), stream up-right otherwise
    n1_odd = n[1, odd, :]
    if len(odd_rows) > 0:
        next_grid[1, even_rows, 1:] = n1_odd[:, :-1]
    if len(odd_rows) > 0:
        next_grid[4, odd_rows, W-1] |= n1_odd[:, -1]

    # d2 (NW): even -> (y-1,x-1), odd -> (y-1,x)
    n2_even = n[2, even, :]
    # Even rows y>0, x>0 -> odd targets (y-1, x-1)
    if len(even_rows) > 1:
        next_grid[2, odd_rows[:-1], 1:] = n2_even[1:, :-1]
    # y=0 (top) bounce to d4
    next_grid[5, 0, :] |= n2_even[0, :]
    # x=0, y>0 (left wall) bounce to d1
    if len(even_rows) > 1:
        next_grid[5, even_rows[1:], 0] |= n2_even[1:, 0]
    # Odd rows: no wall boundaries, stream up
    n2_odd = n[2, odd, :]
    if len(odd_rows) > 0:
        next_grid[2, even_rows, :] = n2_odd

    # d4 (SW): even -> (y+1,x-1), odd -> (y+1,x)
    n4_even = n[4, even, :]
    # x>0 -> odd targets (y+1, x-1)
    if len(even_rows) > 0:
        next_grid[4, odd_rows, :-1] = n4_even[:, 1:]
    # x=0 (left wall) all even rows bounce to d5
    if len(even_rows) > 0:
        next_grid[1, even_rows, 0] |= n4_even[:, 0]
    # Odd rows: bounce bottom (y=H-1), stream down otherwise
    n4_odd = n[4, odd, :]
    if len(odd_rows) > 1:
        next_grid[4, even_rows[1:], :] = n4_odd[:-1, :]
    if len(odd_rows) > 0:
        next_grid[1, H-1, :] |= n4_odd[-1, :]

    # d5 (SE): even -> (y+1,x), odd -> (y+1,x+1)
    n5_even = n[5, even, :]
    # All even rows -> odd targets (y+1, x) — no bounce needed
    if len(even_rows) > 0:
        next_grid[5, odd_rows, :] = n5_even
    # Odd rows: bounce bottom or right, stream down-right otherwise
    n5_odd = n[5, odd, :]
    if len(odd_rows) > 1:
        next_grid[5, even_rows[1:], 1:] = n5_odd[:-1, :-1]
    if len(odd_rows) > 0:
        next_grid[2, odd_rows, W-1] |= n5_odd[:, -1]
    if len(odd_rows) > 0:
        next_grid[1, H-1, :] |= n5_odd[-1, :]

    next_grid[:, barrier] = False

    return next_grid

# --- 4. PLAYBACK STATE ---
state = {
    'running': False,
    'reverse': False,
    'frame':   0,
}

# --- 5. FIGURE LAYOUT ---
fig = plt.figure(figsize=(13, 7), facecolor='#0d0d0d')
fig.patch.set_facecolor('#0d0d0d')

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    height_ratios=[10, 1],
    hspace=0.18,
    wspace=0.08,
    left=0.04, right=0.96,
    top=0.93,  bottom=0.05,
)

ax_sim = fig.add_subplot(gs[0, :])
ax_sim.set_facecolor('#0d0d0d')
ax_sim.set_title(
    "FHP Lattice-Gas  ·  Hexagonal (D2Q6)  ·  Reflective Barrier",
    color='#e0e0e0', fontsize=13, fontweight='bold',
    fontfamily='monospace', pad=8,
)
ax_sim.axis('off')

ax_btn_stop    = fig.add_subplot(gs[1, 0])
ax_btn_reverse = fig.add_subplot(gs[1, 1])
ax_btn_reset   = fig.add_subplot(gs[1, 2])

# --- 6. INITIAL RENDER ---
visual_grid = grid.sum(axis=0).astype(float)
visual_grid[barrier_mask] = 6.0
im = ax_sim.imshow(visual_grid, cmap='inferno', origin='lower', vmin=0, vmax=6,
                   interpolation='nearest', aspect='auto')

frame_text = ax_sim.text(
    0.01, 1.05, 'frame: 0',
    transform=ax_sim.transAxes,
    color='#888888', fontsize=8, fontfamily='monospace',
    va='bottom', clip_on=False,
)

# --- 7. BUTTON STYLING ---
def make_button(ax, label, color_active):
    ax.set_facecolor('#1a1a1a')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
        spine.set_linewidth(1.2)
    btn = Button(ax, label, color='#1a1a1a', hovercolor='#2a2a2a')
    btn.label.set_fontfamily('monospace')
    btn.label.set_fontsize(11)
    btn.label.set_color(color_active)
    btn.label.set_fontweight('bold')
    return btn

btn_stop    = make_button(ax_btn_stop,    '> PLAY',    '#80ee80')
btn_reverse = make_button(ax_btn_reverse, '<< REVERSE', '#60c0f0')
btn_reset   = make_button(ax_btn_reset,   'RESET',   '#f06060')

# --- 8. BUTTON CALLBACKS ---
def on_stop(event):
    state['running'] = not state['running']
    if state['running']:
        btn_stop.label.set_text('|| PAUSE')
        btn_stop.label.set_color('#f0c040')
    else:
        btn_stop.label.set_text('> PLAY')
        btn_stop.label.set_color('#80ee80')
    fig.canvas.draw_idle()

def on_reverse(event):
    if not history:
        return
    state['reverse'] = not state['reverse']
    if state['reverse']:
        btn_reverse.label.set_text('>> FORWARD')
        btn_reverse.label.set_color('#f0a020')
    else:
        btn_reverse.label.set_text('<< REVERSE')
        btn_reverse.label.set_color('#60c0f0')
    fig.canvas.draw_idle()

def on_reset(event):
    global grid, history
    grid = init_grid()
    history.clear()
    state['running'] = False
    state['reverse'] = False
    state['frame']   = 0
    btn_stop.label.set_text('> PLAY')
    btn_stop.label.set_color('#80ee80')
    btn_reverse.label.set_text('<< REVERSE')
    btn_reverse.label.set_color('#60c0f0')
    density = grid.sum(axis=0).astype(float)
    density[barrier_mask] = 6.0
    im.set_array(density)
    frame_text.set_text('frame:     0')
    fig.canvas.draw_idle()

btn_stop.on_clicked(on_stop)
btn_reverse.on_clicked(on_reverse)
btn_reset.on_clicked(on_reset)

# --- 9. ANIMATION UPDATE ---
def update(frame_ignored):
    global grid, history

    if not state['running']:
        return [im, frame_text]

    if state['reverse']:
        if history:
            grid = history.pop()
            state['frame'] = max(0, state['frame'] - 1)
        else:
            state['reverse'] = False
            state['running'] = False
            btn_reverse.label.set_text('<< REVERSE')
            btn_reverse.label.set_color('#60c0f0')
            btn_stop.label.set_text('> PLAY')
            btn_stop.label.set_color('#80ee80')
    else:
        history.append(grid.copy())
        grid = step_fhp(grid, barrier_mask, left_face, right_face)
        state['frame'] += 1

    density = grid.sum(axis=0).astype(float)
    density[barrier_mask] = 6.0
    im.set_array(density)
    frame_text.set_text(f"frame: {state['frame']:>5d}")
    return [im, frame_text]

ani = animation.FuncAnimation(
    fig, update, frames=STEPS, interval=20, blit=False,
)

try:
    from IPython.display import HTML, display
    html_video = HTML(ani.to_jshtml())
    display(html_video)
except ImportError:
    plt.show()
