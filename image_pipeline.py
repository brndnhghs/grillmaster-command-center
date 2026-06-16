#!/usr/bin/env python3
"""Backward-compatible wrapper — delegates to the v2 modular pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + ":" + os.environ.get("PYTHONPATH", "")
# Re-exec with proper module invocation
cmd = [sys.executable, "-m", "image_pipeline.pipeline"] + sys.argv[1:]
os.execvp(sys.executable, cmd)