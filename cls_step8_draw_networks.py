import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
import re

model = 'ent'

# ==================== 1. Configuration and data loading ====================
# YEO 17+1 network names and color mapping (including N00 Subcortical)
YEO_17_INFO = {
    0: ('Subcortical', '#808080'),  # Subcortical network, gray
    1: ('VisPeri', '#751975'), 2: ('VisCent', '#359a94'), 3: ('SomMotA', '#009944'), 
    4: ('SomMotB', '#00d9c1'), 5: ('DorsAttnA', '#336600'), 6: ('DorsAttnB', '#659900'), 
    7: ('SalVentAttnA', '#d32910'), 8: ('SalVentAttnB', '#f08e01'), 9: ('Limbic1', '#e3b2f1'), 
    10: ('Limbic2', '#ffab00'), 11: ('ContC', '#996633'), 12: ('ContA', '#f190b7'), 
    13: ('ContB', '#f5d353'), 14: ('DefaultD', '#cc6699'), 15: ('DefaultC', '#ff6666'), 
    16: ('DefaultA', '#cc0033'), 17: ('DefaultB', '#990033')
}

csv_path = f'freq1_diff-{model}.csv'  # or 'freq1_diff.csv', depending on the actual file name
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"File not found: {csv_path}")
    exit()

# Column mapping: use p_fdr for filtering, relaxed to p < 0.05 to include borderline significant features
df_sig = df[df['p_fdr'] < 0.05].copy()
print(f"Total significant features with FDR < 0.05: {len(df_sig)}")

# ==================== 2. Aggregate connection data (dimension reduction) ====================
def parse_network_id(feature_name):
    """Parse network IDs from a feature name, e.g. N01_N02_max -> (1, 2)"""
    match = re.match(r'^N(\d+)_(?:N(\d+)|intra)_', feature_name)
    if match:
        id1 = int(match.group(1))
        id2 = int(match.group(2)) if match.group(2) else id1
        return id1, id2
    return None, None

edges_agg = {}
intra_agg = {}

for _, row in df_sig.iterrows():
    feat_name = row['Feature_CSV']  # Use the Feature_CSV column
    id1, id2 = parse_network_id(feat_name)
    if id1 is None: 
        continue

    # Compute mean difference: ISM_Mean - HC_Mean
    # Positive indicates ISM > HC (red), negative indicates ISM < HC (blue)
    diff = row['ISM_Mean'] - row['HC_Mean']
    weight = 1 

    if id1 == id2:
        if id1 not in intra_agg:
            intra_agg[id1] = {'increase': 0, 'decrease': 0, 'total': 0}
        intra_agg[id1]['total'] += weight
        if diff > 0:
            intra_agg[id1]['increase'] += weight
        else:
            intra_agg[id1]['decrease'] += weight
    else:
        n_min, n_max = min(id1, id2), max(id1, id2)
        key = (n_min, n_max)

        if key not in edges_agg:
            edges_agg[key] = {'increase': 0, 'decrease': 0, 'total': 0}
        edges_agg[key]['total'] += weight
        if diff > 0:
            edges_agg[key]['increase'] += weight
        else:
            edges_agg[key]['decrease'] += weight

print(f"Inter-network connections: {len(edges_agg)} pairs, intra-network connections: {len(intra_agg)} networks")

# ==================== 3. Circular layout plotting ====================
fig, ax = plt.subplots(figsize=(14, 14))
ax.set_aspect('equal')
ax.axis('off')

n_nodes = 18  # 18 nodes in total (0-17)
angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
angles = np.pi/2 - angles  # Start from the top

radius = 10.0
node_radius = 0.9

node_positions = {}
for i, angle in enumerate(angles):
    node_id = i  # Starting from 0
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    node_positions[node_id] = (x, y)

# ==================== 4. Draw inter-network connections ====================
def draw_bezier_curve(ax, p1, p2, color, linewidth, alpha):
    """Draw a Bezier curve connecting two nodes"""
    x1, y1 = p1
    x2, y2 = p2
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ctrl_x, ctrl_y = mid_x * 0.6, mid_y * 0.6

    verts = [(x1, y1), (ctrl_x, ctrl_y), (x2, y2)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)

    patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color, 
                               linewidth=linewidth, alpha=alpha, zorder=2)
    ax.add_patch(patch)

for (n1, n2), stats in edges_agg.items():
    p1 = node_positions[n1]
    p2 = node_positions[n2]

    if stats['increase'] > stats['decrease']:
        color = '#d62728'  # Red: ISM > HC
    elif stats['decrease'] > stats['increase']:
        color = '#1f77b4'  # Blue: ISM < HC
    else:
        color = 'gray'

    if stats['total'] == 1:
        lw = 2.0
    elif stats['total'] == 2:
        lw = 4.5
    else:
        lw = 7.0

    draw_bezier_curve(ax, p1, p2, color, lw, alpha=0.7)

# ==================== 5. Draw nodes and self-loops (halos) ====================
for node_id, (x, y) in node_positions.items():
    net_name, net_color = YEO_17_INFO[node_id]

    # Draw the halo for intra-network connections
    if node_id in intra_agg:
        stats = intra_agg[node_id]
        if stats['increase'] > stats['decrease']:
            halo_color = '#d62728'
        elif stats['decrease'] > stats['increase']:
            halo_color = '#1f77b4'
        else:
            halo_color = 'gray'

        halo_lw = 3.0 + stats['total'] * 2.5 

        halo = plt.Circle((x, y), node_radius + 0.35, 
                          facecolor='none', 
                          edgecolor=halo_color, 
                          linewidth=halo_lw, 
                          alpha=0.8, zorder=4)
        ax.add_patch(halo)

    # Draw the node circle
    circle = plt.Circle((x, y), node_radius, 
                         facecolor=net_color, 
                         edgecolor='black', 
                         linewidth=2.0, 
                         zorder=5)
    ax.add_patch(circle)

    # Node ID text
    ax.text(x, y, f"{node_id}", 
            ha='center', va='center', 
            fontsize=15, fontweight='bold', color='white',
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')],
            zorder=6)

    # Network label
    angle = np.arctan2(y, x)
    label_dist = radius + node_radius + 0.6
    label_x = label_dist * np.cos(angle)
    label_y = label_dist * np.sin(angle)

    ha = 'left' if np.cos(angle) >= 0 else 'right'

    ax.text(label_x, label_y, net_name, 
            ha=ha, va='center', 
            fontsize=14, fontweight='bold', color=net_color)

# ==================== 6. Merge legend ====================
legend_handles = [
    # --- Direction group ---
    mpatches.Patch(color='none', label='Direction of Alteration'),
    plt.Line2D([0], [0], color='#d62728', linewidth=4, label='  ISM > HC (Hyper-connectivity)'),
    plt.Line2D([0], [0], color='#1f77b4', linewidth=4, label='  ISM < HC (Hypo-connectivity)'),
    # Blank separator
    mpatches.Patch(color='none', label=' '),
    # --- Evidence strength group ---
    mpatches.Patch(color='none', label='Evidence Strength (Line Width / Halo)'),
    plt.Line2D([0], [0], color='black', linewidth=2.0, label='  1 Significant Feature'),
    plt.Line2D([0], [0], color='black', linewidth=4.5, label='  2 Significant Features'),
    plt.Line2D([0], [0], color='black', linewidth=7.0, label='  3+ Significant Features'),
]

leg = ax.legend(handles=legend_handles, 
                loc='upper right', 
                bbox_to_anchor=(1.10, 1.03),
                fontsize=11, 
                frameon=False, framealpha=0.95, edgecolor='gray')

# Make the group headers bold
for text in leg.get_texts():
    label = text.get_text().strip()
    if label in ['Direction of Alteration', 'Evidence Strength (Line Width / Halo)']:
        text.set_fontweight('bold')
        text.set_fontsize(12)

# Explanatory note
ax.text(0.5, 0.05, 
        "Note: Lines = inter-network; Halos = intra-network alterations.", 
        transform=ax.transAxes, ha='center', fontsize=11, style='italic', color='gray')

ax.set_title("Network-Level Alterations\n"
             "in Insomnia Disorder vs Healthy Controls", 
             fontsize=18, fontweight='bold', pad=30)

margin = 5
ax.set_xlim(-radius - margin, radius + margin)
ax.set_ylim(-radius - margin, radius + margin)

plt.tight_layout()

# Ensure the legend is not cropped when saving
plt.savefig(f'Circular_Connectogram_Simplified-{model}.pdf', dpi=300, 
            bbox_inches='tight', facecolor='white',
            bbox_extra_artists=[leg], format='pdf')
plt.show()

