import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from scipy import signal

class AMModulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AM Modulation & Demodulation Simulator")
        self.root.geometry("1400x900")
        
        # Default parameters
        self.message_freq = 5  # Hz
        self.carrier_freq = 100  # Hz
        self.modulation_index = 0.8
        self.sampling_rate = 2000  # Hz
        self.duration = 1  # seconds
        
        self.setup_ui()
        self.update_plots()
    
    def setup_ui(self):
        # Control Frame
        control_frame = ttk.LabelFrame(self.root, text="Control Parameters", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Message Frequency
        ttk.Label(control_frame, text="Message Frequency (Hz):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.msg_freq_var = tk.DoubleVar(value=self.message_freq)
        msg_freq_slider = ttk.Scale(control_frame, from_=1, to=20, variable=self.msg_freq_var, 
                                     orient=tk.HORIZONTAL, length=200, command=self.on_slider_change)
        msg_freq_slider.grid(row=0, column=1, pady=5)
        self.msg_freq_label = ttk.Label(control_frame, text=f"{self.message_freq:.1f} Hz")
        self.msg_freq_label.grid(row=0, column=2, pady=5)
        
        # Carrier Frequency
        ttk.Label(control_frame, text="Carrier Frequency (Hz):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.carrier_freq_var = tk.DoubleVar(value=self.carrier_freq)
        carrier_freq_slider = ttk.Scale(control_frame, from_=50, to=200, variable=self.carrier_freq_var,
                                         orient=tk.HORIZONTAL, length=200, command=self.on_slider_change)
        carrier_freq_slider.grid(row=1, column=1, pady=5)
        self.carrier_freq_label = ttk.Label(control_frame, text=f"{self.carrier_freq:.1f} Hz")
        self.carrier_freq_label.grid(row=1, column=2, pady=5)
        
        # Modulation Index
        ttk.Label(control_frame, text="Modulation Index:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.mod_index_var = tk.DoubleVar(value=self.modulation_index)
        mod_index_slider = ttk.Scale(control_frame, from_=0.1, to=1.5, variable=self.mod_index_var,
                                      orient=tk.HORIZONTAL, length=200, command=self.on_slider_change)
        mod_index_slider.grid(row=2, column=1, pady=5)
        self.mod_index_label = ttk.Label(control_frame, text=f"{self.modulation_index:.2f}")
        self.mod_index_label.grid(row=2, column=2, pady=5)
        
        # Update Button
        ttk.Button(control_frame, text="Update Plots", command=self.update_plots).grid(row=3, column=0, columnspan=3, pady=20)
        
        # Information
        info_text = """
AM Modulation Info:
━━━━━━━━━━━━━━━━━━
• Modulation Index (μ):
  - μ < 1: Under-modulation
  - μ = 1: Critical modulation
  - μ > 1: Over-modulation

• Formula:
  s(t) = Ac[1 + μ·m(t)]·cos(2πfc·t)
  
• Demodulation uses:
  - Envelope detection
  - Rectification + Low-pass filter
        """
        info_label = ttk.Label(control_frame, text=info_text, justify=tk.LEFT, 
                               font=('Courier', 9), background='#f0f0f0')
        info_label.grid(row=4, column=0, columnspan=3, pady=10, sticky=tk.W)
        
        # Plot Frame
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure with subplots
        self.fig = Figure(figsize=(12, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def on_slider_change(self, event=None):
        self.msg_freq_label.config(text=f"{self.msg_freq_var.get():.1f} Hz")
        self.carrier_freq_label.config(text=f"{self.carrier_freq_var.get():.1f} Hz")
        self.mod_index_label.config(text=f"{self.mod_index_var.get():.2f}")
    
    def generate_signals(self):
        # Time array
        t = np.linspace(0, self.duration, int(self.sampling_rate * self.duration))
        
        # Message signal (baseband)
        fm = self.msg_freq_var.get()
        message_signal = np.cos(2 * np.pi * fm * t)
        
        # Carrier signal
        fc = self.carrier_freq_var.get()
        carrier_signal = np.cos(2 * np.pi * fc * t)
        
        # AM modulated signal
        mu = self.mod_index_var.get()
        am_signal = (1 + mu * message_signal) * carrier_signal
        
        # Demodulation
        # Rectification
        rectified = np.abs(am_signal)
        
        # Low-pass filter (envelope detection)
        cutoff_freq = 2 * fm  # Cutoff at twice the message frequency
        sos = signal.butter(4, cutoff_freq, 'low', fs=self.sampling_rate, output='sos')
        demodulated = signal.sosfilt(sos, rectified)
        
        # Remove DC component and normalize
        demodulated = demodulated - np.mean(demodulated)
        demodulated = demodulated / np.max(np.abs(demodulated)) if np.max(np.abs(demodulated)) > 0 else demodulated
        
        return t, message_signal, carrier_signal, am_signal, rectified, demodulated
    
    def update_plots(self):
        self.fig.clear()
        
        t, msg, carrier, am, rectified, demod = self.generate_signals()
        
        # Limit display time for clarity
        display_time = min(0.2, self.duration)
        display_samples = int(display_time * self.sampling_rate)
        
        # Plot 1: Message Signal
        ax1 = self.fig.add_subplot(5, 1, 1)
        ax1.plot(t[:display_samples], msg[:display_samples], 'b-', linewidth=2)
        ax1.set_ylabel('Amplitude', fontsize=10)
        ax1.set_title('Message Signal (Baseband)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, display_time)
        
        # Plot 2: Carrier Signal
        ax2 = self.fig.add_subplot(5, 1, 2)
        ax2.plot(t[:display_samples], carrier[:display_samples], 'r-', linewidth=1)
        ax2.set_ylabel('Amplitude', fontsize=10)
        ax2.set_title('Carrier Signal', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, display_time)
        
        # Plot 3: AM Modulated Signal
        ax3 = self.fig.add_subplot(5, 1, 3)
        ax3.plot(t[:display_samples], am[:display_samples], 'g-', linewidth=1)
        ax3.plot(t[:display_samples], (1 + self.mod_index_var.get() * msg[:display_samples]), 'k--', 
                 linewidth=1.5, alpha=0.6, label='Envelope')
        ax3.plot(t[:display_samples], -(1 + self.mod_index_var.get() * msg[:display_samples]), 'k--', 
                 linewidth=1.5, alpha=0.6)
        ax3.set_ylabel('Amplitude', fontsize=10)
        ax3.set_title(f'AM Modulated Signal (μ = {self.mod_index_var.get():.2f})', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, display_time)
        
        # Plot 4: Rectified Signal
        ax4 = self.fig.add_subplot(5, 1, 4)
        ax4.plot(t[:display_samples], rectified[:display_samples], 'm-', linewidth=1)
        ax4.set_ylabel('Amplitude', fontsize=10)
        ax4.set_title('Rectified Signal (Envelope Detection Step 1)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, display_time)
        
        # Plot 5: Demodulated Signal
        ax5 = self.fig.add_subplot(5, 1, 5)
        ax5.plot(t[:display_samples], demod[:display_samples], 'c-', linewidth=2, label='Demodulated')
        ax5.plot(t[:display_samples], msg[:display_samples], 'b--', linewidth=1.5, alpha=0.6, label='Original')
        ax5.set_xlabel('Time (seconds)', fontsize=10)
        ax5.set_ylabel('Amplitude', fontsize=10)
        ax5.set_title('Demodulated Signal (After Low-Pass Filter)', fontsize=12, fontweight='bold')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, display_time)
        
        self.fig.tight_layout()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = AMModulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
