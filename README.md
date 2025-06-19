# ComfyUI Pipeline Manager + External Checks

This project provides a web interface (built with **Gradio**) that orchestrates two independent ComfyUI *workflows* while running external checks (NSFW detection, face counting, partial-face detection, …) between them.

---

## ❱ Requirements

1. **Python 3.10 – 3.12** (use `pyenv`, `conda` or your favourite manager).
2. **ComfyUI** running in REST-server mode on **port 8188**.
3. A NVIDIA GPU + CUDA is highly recommended but not mandatory (for PyTorch / ComfyUI acceleration).
4. System libraries required by OpenCV / Pillow (Ubuntu example):
   ```bash
   sudo apt update && sudo apt install -y libgl1 libglib2.0-0
   ```

---

## 🔧 Setting-up the Python environment

```bash
# 1) Clone or copy the repository
cd /path/to/project

# 2) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4) Install PyTorch manually according to your GPU / CUDA version
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> **Note**: *requirements.txt* **does not include** PyTorch so that you can pick the appropriate build (CPU-only, CUDA 12, etc.). For CPU-only simply run `pip install torch torchvision`.

---

## ⚙️ Installing ComfyUI

```bash
# Clone ComfyUI (skip if you already have it)
cd $HOME
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install ComfyUI dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# (Optional) install PyTorch in this environment if you haven't done it before
```

### Running ComfyUI in server mode

The orchestrator talks to ComfyUI via its REST API. Make sure ComfyUI is running **before** launching the GUI. Do **not** manually load the workflows — they will be posted automatically.

```bash
python main.py --listen 0.0.0.0 --port 8188
```

If ComfyUI is running elsewhere, set the `COMFY_HOST` environment variable **before** starting the GUI:

```bash
export COMFY_HOST="http://192.168.1.50:8189"
```

---

## 📂 Relevant folders & files

| Path | Description |
|------|-------------|
| `Workflow1.json` | Generates the initial image. |
| `Workflow2.json` | Post-processes the image **only if** it passes all checks. |
| `Nodes/` | facexlib weights and auxiliary models used by the checks. |
| `output/` | (external) ComfyUI output folder. Auto-detected at `$HOME/ComfyUI/output`, override with `COMFY_OUTPUT_DIR`. |

---

## ▶️ Launching the Gradio interface

With ComfyUI already running, open a second terminal inside the project folder and run:

```bash
python orchestrator_gui.py
```

By default the Gradio server listens on `http://localhost:18188` and is shared to your LAN (`--share`).

### Interface elements

1. **Run Pipeline** – start the process.
2. **In Queue** – fixed box (top-right) showing how many successful images are still pending.
3. **Checks to Perform** – enable NSFW, face counting, partial-face detection, …
4. **Workflow 1 / Workflow 2 Config** – tabs that let you edit any node parameter before execution.
5. **Log** – real-time console.
6. **Results** – gallery with approved final images.

---

## ✅ Batch mode

The *"Number of runs"* slider sets how many **successful** images you want. The orchestrator keeps generating attempts (incrementing the seed) until that number is reached. The "In Queue" counter updates live.

---

## 📑 Useful environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `COMFY_HOST` | `http://127.0.0.1:8188` | Base URL of the ComfyUI REST server. |
| `COMFY_OUTPUT_DIR` | Auto-detected (`$HOME/ComfyUI/output`) | Where ComfyUI saves PNGs. |

---

## 🛠️ Helper scripts

```bash
./setup_env.sh   # example script for Linux to install system deps
```

---

## ℹ️ FAQ

**The "In Queue" counter doesn't update.** Make sure you are running a version of `orchestrator_gui.py` that includes the *pending-box* feature and check the console for errors.

**Can I run the GUI on Windows while ComfyUI runs on a Linux PC?** Yes — just set `COMFY_HOST` to the remote machine's public IP and open port 8188 on the firewall.

---

## License

This project is released under the MIT License (see `LICENSE` if provided). 