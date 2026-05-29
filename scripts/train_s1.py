#!/usr/bin/env python3
"""[DEPRECATED] → use: python scripts/train_stage.py --stage S1"""

import subprocess, sys
sys.exit(subprocess.run([sys.executable, "scripts/train_stage.py", "--stage", "S1", *sys.argv[1:]]).returncode)
