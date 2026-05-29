#!/usr/bin/env python3
"""S3 wrapper → python scripts/train_stage.py --stage S3"""

import subprocess, sys
sys.exit(subprocess.run([sys.executable, "scripts/train_stage.py", "--stage", "S3", *sys.argv[1:]]).returncode)
