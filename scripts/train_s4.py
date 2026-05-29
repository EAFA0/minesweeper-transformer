#!/usr/bin/env python3
"""S4 wrapper → python scripts/train_stage.py --stage S4"""

import subprocess, sys
sys.exit(subprocess.run([sys.executable, "scripts/train_stage.py", "--stage", "S4", *sys.argv[1:]]).returncode)
