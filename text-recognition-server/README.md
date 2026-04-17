# Run the Application

Use these steps from the project root directory.

## 1. Create a virtual environment

```bash
python -m venv venv
```

## 2. Activate the virtual environment

Linux/macOS:

```bash
source venv/bin/activate
```

Windows (PowerShell):

```powershell
venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```bat
venv\Scripts\activate.bat
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Clone YOLOv5

Your setup requires YOLOv5 source files and they are not already present, run:

```bash
git clone https://github.com/ultralytics/yolov5
```

## 5. Start the GUI application

```bash
python full_pipeline_gui_new.py
```

## Quick troubleshooting

- If `python` is not found, try `python3`.
- If packages fail to import, confirm the virtual environment is activated before running the app.