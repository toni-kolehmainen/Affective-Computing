# Import packages to trigger __init__.py
import src.ui
from src.ui.main import main as ui_main
import linear_eval

if hasattr(src.ui.main, "main"):
    ui_main()

if hasattr(linear_eval, "main"):
    linear_eval.main()
