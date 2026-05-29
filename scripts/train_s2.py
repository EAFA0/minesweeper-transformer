#!/usr/bin/env python3
"""S2 wrapper → python scripts/train_stage.py --stage S2"""

import subprocess, sys
sys.exit(subprocess.run([sys.executable, "scripts/train_stage.py", "--stage", "S2", *sys.argv[1:]]).returncode)
