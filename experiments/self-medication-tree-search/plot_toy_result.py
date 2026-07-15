import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = 0.83
data = [
    ("baseline", BASE),
    ("focused", 1.00), ("curious", 1.00), ("anxious", 1.00), ("dumbed_down", 1.00), ("calm", 0.83),
    ("focused + curious", 1.00), ("focused + anxious", 1.00), ("curious + anxious", 1.00),
    ("anxious + dumbed_down", 1.00), ("curious + calm", 1.00),
    ("focused + dumbed_down", 0.83),
    ("focused + calm", 0.67), ("anxious + calm", 0.67),
    ("curious + dumbed_down", 0.50),
]
data.sort(key=lambda x: (x[1], x[0] != "baseline"))  # low->high, baseline distinct
labels = [d[0] for d in data]
vals = [d[1] for d in data]

def color(name, v):
    if name == "baseline": return "#6E86B8"      # steel
    if v < BASE - 1e-9:     return "#C25B4E"      # drop  (red)
    if v > BASE + 1e-9:     return "#7BB662"      # gain  (green)
    return "#B9A96B"                               # tie   (muted gold)

fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=200)
y = range(len(labels))
ax.barh(list(y), vals, color=[color(n, v) for n, v in zip(labels, vals)], height=0.72, edgecolor="none")
ax.axvline(BASE, color="#6E86B8", ls="--", lw=1.4)
ax.text(BASE, len(labels) - 0.2, f"  baseline {BASE:.2f}", color="#42506b", fontsize=9, va="bottom", fontweight="bold")
for i, v in enumerate(vals):
    ax.text(v - 0.015, i, f"{v:.2f}", va="center", ha="right", color="white", fontsize=8.5, fontweight="bold")
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlim(0, 1.06); ax.set_xlabel("task accuracy  (6 GSM8K-style problems)", fontsize=10)
ax.set_title("Tree-search over steered self-states — 6-compound toy run\nQwen3-8B (MLX), beam w2 d2 · benchmark saturated",
             fontsize=12.5, fontweight="bold", loc="left", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=0)
fig.text(0.045, -0.005,
         "Singles ceiling (~1.0) → benchmark too easy;  the only movement is 2-compound states that DROP (interaction / over-steering).\n"
         "Negative control  dumbed_down = 1.00  (did NOT drop) → metric non-discriminative → moved to GPQA (~40%, headroom).",
         fontsize=8.6, color="#555", ha="left")
plt.tight_layout()
plt.savefig("outputs/toy_6compound.png", bbox_inches="tight", facecolor="white")
print("saved outputs/toy_6compound.png")
