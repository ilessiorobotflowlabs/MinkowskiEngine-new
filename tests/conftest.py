import sys
from pathlib import Path

# Add tests/python to path for utils.gradcheck imports
sys.path.insert(0, str(Path(__file__).parent / "python"))
