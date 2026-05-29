#!/usr/bin/env python3
"""S1.5 wrapper → python scripts/train_stage.py --stage S1.5"""

import subprocess, sys
sys.exit(subprocess.run([sys.executable, "scripts/train_stage.py", "--stage", "S1.5", *sys.argv[1:]]).returncode)
