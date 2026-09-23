#!/usr/bin/env python

import sys

from run_study2_pilot import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/study2_zero_noise.yaml"])
    main()
