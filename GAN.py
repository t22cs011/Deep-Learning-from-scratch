import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm # For basic font access

print("Generating conceptual graph for ideal D output probabilities (English labels).")
plt.style.use('seaborn-v0_8-whitegrid')

# --- Conceptual Data for Ideal GAN Training ---
iterations_conceptual = np.linspace(0, 100, 100) # Representing 100 steps/epochs of training

# D(Real Data) Output: Starts lower, quickly rises to ~0.95 and stabilizes
d_real_ideal = np.ones_like(iterations_conceptual) * 0.95
# Initial rise phase
initial_rise_length = 20
d_real_ideal[:initial_rise_length] = np.linspace(0.6, 0.95, initial_rise_length)
# Add a little bit of noise for realism, but keep it stable high
d_real_ideal[initial_rise_length:] += np.random.normal(0, 0.01, 100 - initial_rise_length)
d_real_ideal = np.clip(d_real_ideal, 0, 1.0)

# D(G's Output) Output: Starts at 0.5, D learns (drops), G learns (rises), stabilizes around 0.5
d_fake_ideal = np.zeros_like(iterations_conceptual)
# Phase 1: Initial (D unsure)
phase1_end = 10
d_fake_ideal[:phase1_end] = np.linspace(0.5, 0.5, phase1_end)
# Phase 2: D learns to detect fakes (drops)
phase2_end = 30
d_fake_ideal[phase1_end-1:phase2_end] = np.linspace(0.5, 0.1, phase2_end - (phase1_end-1))
# Phase 3: G learns to fool D (rises)
phase3_end = 60
d_fake_ideal[phase2_end-1:phase3_end] = np.linspace(0.1, 0.6, phase3_end - (phase2_end-1)) # Overshoots 0.5 slightly then settles
# Phase 4: Stabilizes around 0.5 with minor oscillations
stable_phase_length = 100 - (phase3_end-1)
time_for_oscillation = np.linspace(0, 4 * np.pi, stable_phase_length)
oscillations = 0.05 * np.sin(time_for_oscillation)
d_fake_ideal[phase3_end-1:] = 0.5 + oscillations + np.random.normal(0, 0.02, stable_phase_length)
d_fake_ideal = np.clip(d_fake_ideal, 0, 1.0)


# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6)) # Single plot

ax.plot(iterations_conceptual, d_real_ideal, label='Avg D(Real Data) Output (Ideal)', color='royalblue', linewidth=2)
ax.plot(iterations_conceptual, d_fake_ideal, label="Avg D(G's Output) Output (Ideal)", color='darkorange', linewidth=2)

ax.set_xlabel('Training Progress (Conceptual Iterations)', fontsize=12)
ax.set_ylabel('Avg. Discriminator Output Probability', fontsize=12)
ax.set_title('Ideal GAN: Discriminator Output Probabilities', fontsize=14, pad=15)

ax.legend(fontsize=10)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.set_ylim(-0.05, 1.05)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(pad=1.0)
plt.show()