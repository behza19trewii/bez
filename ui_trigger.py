import tkinter as tk
from tkinter import filedialog, messagebox

def get_user_config():
    root = tk.Tk()
    root.title("Initial User Configuration Trigger")
    root.geometry("420x550")
    root.resizable(False, False)

    inputs = {}

    def add_field(label, default):
        frame = tk.Frame(root)
        frame.pack(pady=4, fill="x", padx=20)
        tk.Label(frame, text=label, width=20, anchor="w").pack(side="left")
        var = tk.StringVar(value=str(default))
        entry = tk.Entry(frame, textvariable=var, width=20)
        entry.pack(side="right")
        inputs[label] = var

    def select_file():
        file_path = filedialog.askopenfilename(title="Select Data File", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            inputs["Data Path"].set(file_path)

    # ---- fields ----
    tk.Label(root, text="📊 SMA Strategy Configuration", font=("Arial", 12, "bold")).pack(pady=10)

    add_field("MA_SHORT_DEFAULT", 10)
    add_field("MA Long", 20)
    add_field("Initial Capital", 10000.0)
    add_field("Position Size %", 0.5)
    add_field("Commission", 0.0)
    add_field("Slippage", 0.0005)
    add_field("Exit on MA Cross", True)
    add_field("Take Profit %", 0.05)
    add_field("Stop Loss %", -0.05)
    add_field("Max Holding Days", "")
    add_field("Use Trailing ATR", True)
    add_field("ATR Mult", 3.0)
    add_field("ATR Period", 14)

    # ---- file select ----
    frame_file = tk.Frame(root)
    frame_file.pack(pady=8, fill="x", padx=20)
    tk.Label(frame_file, text="Data Path", width=20, anchor="w").pack(side="left")
    inputs["Data Path"] = tk.StringVar()
    entry_file = tk.Entry(frame_file, textvariable=inputs["Data Path"], width=20)
    entry_file.pack(side="left", padx=4)
    tk.Button(frame_file, text="Select from computer", command=select_file).pack(side="right")

    # ---- submit ----
    def on_submit():
        try:
            config = {
                "MA_SHORT_DEFAULT": int(inputs["MA_SHORT_DEFAULT"].get()),
                "ma_long": int(inputs["MA Long"].get()),
                "initial_capital": float(inputs["Initial Capital"].get()),
                "position_size_pct": float(inputs["Position Size %"].get()),
                "commission": float(inputs["Commission"].get()),
                "slippage": float(inputs["Slippage"].get()),
                "exit_on_ma_cross": inputs["Exit on MA Cross"].get().lower() in ["true", "1", "yes"],
                "take_profit_pct": float(inputs["Take Profit %"].get()),
                "stop_loss_pct": float(inputs["Stop Loss %"].get()),
                "max_holding_days": int(inputs["Max Holding Days"].get()) if inputs["Max Holding Days"].get() else None,
                "use_trailing_atr": inputs["Use Trailing ATR"].get().lower() in ["true", "1", "yes"],
                "atr_mult": float(inputs["ATR Mult"].get()),
                "atr_period": int(inputs["ATR Period"].get()),
                "data_path": inputs["Data Path"].get()
            }
            if not config["data_path"]:
                messagebox.showerror("Error", "Please select a data file before running.")
                return
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Error in input values: {e}")
            return

        root.destroy()
        global final_config
        final_config = config

    tk.Button(root, text="Run Strategy", bg="#4CAF50", fg="white", command=on_submit).pack(pady=15)
    tk.Label(root, text="Developed for SMA Strategy ⚙️", fg="gray").pack(pady=5)

    root.mainloop()
    return final_config
