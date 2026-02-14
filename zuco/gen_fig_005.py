import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- Data from loss_history.txt files ---

# (1) Full Curriculum -> ZuCo 1.0
full_curr_zuco1 = {
    'epochs': [1, 2, 3, 4, 5, 6],
    'train':  [0.223671, 0.121991, 0.112082, 0.082952, 0.041160, 0.020572],
    'val':    [0.132026, 0.152305, 0.159835, 0.301764, 0.585506, 0.694644],
    'best_epoch': 1,  # lowest val loss at epoch 1
}

# (2) Sleep -> ZuCo 2.0
sleep_zuco2 = {
    'epochs': [1, 2, 3, 4, 5, 6, 7, 8],
    'train':  [0.405719, 0.159731, 0.155235, 0.149044, 0.127732, 0.080260, 0.046015, 0.035494],
    'val':    [0.201449, 0.201504, 0.185276, 0.259822, 0.457372, 0.528631, 0.680275, 0.813444],
    'best_epoch': 3,  # lowest val loss at epoch 3
}

# (3) Scratch -> ZuCo 2.0
scratch_zuco2 = {
    'epochs': [1, 2, 3],
    'train':  [0.242099, 0.180254, 0.175301],
    'val':    [0.156800, 0.144128, 0.174064],
    'best_epoch': 2,  # lowest val loss at epoch 2
}

# (4) HAR -> ZuCo 1.0
har_zuco1 = {
    'epochs': [1, 2, 3, 4, 5, 6, 7],
    'train':  [0.482374, 0.159083, 0.152620, 0.139432, 0.108440, 0.061137, 0.040128],
    'val':    [0.246200, 0.143546, 0.178345, 0.307677, 0.482211, 0.536223, 0.874739],
    'best_epoch': 2,  # lowest val loss at epoch 2
}

experiments = [
    ('Full Curriculum \u2192 ZuCo 1.0', full_curr_zuco1),
    ('Sleep \u2192 ZuCo 2.0', sleep_zuco2),
    ('Scratch \u2192 ZuCo 2.0', scratch_zuco2),
    ('HAR \u2192 ZuCo 1.0', har_zuco1),
]

# --- Color palette ---
train_color = '#2171B5'   # blue
val_color = '#CB181D'     # red
best_color = '#FFC107'    # gold star

# --- Figure ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor='white')
fig.suptitle('Training Loss Curves: Rapid Overfitting Pattern',
             fontsize=14, fontweight='bold', y=0.97)

for ax, (title, data) in zip(axes.flat, experiments):
    epochs = data['epochs']
    train = data['train']
    val = data['val']
    best_ep = data['best_epoch']
    best_val = val[best_ep - 1]

    # Plot lines
    ax.plot(epochs, train, '-o', color=train_color, linewidth=2,
            markersize=5, label='Train loss', zorder=3)
    ax.plot(epochs, val, '--s', color=val_color, linewidth=2,
            markersize=5, label='Val loss', zorder=3)

    # Star marker on best validation epoch
    ax.plot(best_ep, best_val, marker='*', color=best_color,
            markersize=18, markeredgecolor='black', markeredgewidth=0.8,
            zorder=5, label=f'Best val (epoch {best_ep})')

    # Styling
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_xticks(epochs)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.3)

    # Light shading for overfitting region (after best epoch)
    if best_ep < epochs[-1]:
        ax.axvspan(best_ep + 0.5, epochs[-1] + 0.3, alpha=0.07,
                   color='red', zorder=0)
        # Small annotation
        mid_x = (best_ep + epochs[-1]) / 2 + 0.3
        y_top = ax.get_ylim()[1]
        ax.text(mid_x, y_top * 0.88, 'overfitting',
                fontsize=8, color='#999999', ha='center', style='italic')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/home/wangni/notion-figures/zuco/fig_005.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("Figure saved to /home/wangni/notion-figures/zuco/fig_005.png")
