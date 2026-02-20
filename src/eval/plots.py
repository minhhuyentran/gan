import os
import matplotlib.pyplot as plt

def save_hist_compare(real_vals, synth_vals, title: str, out_path: str, bins=50):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.hist(real_vals, bins=bins, alpha=0.5, label="real")
    plt.hist(synth_vals, bins=bins, alpha=0.5, label="synth")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
