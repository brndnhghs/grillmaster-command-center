import sys
sys.path.insert(0, ".")
from image_pipeline.core.registry import get_all
import image_pipeline.methods  # trigger auto-import
all_methods = get_all()
if "1007" in all_methods:
    m = all_methods["1007"]
    print(f"FOUND: id={m.id} name={m.name} category={m.category}")
    print(f"  outputs={m.outputs}")
    print(f"  params={list(m.params.keys()) if hasattr(m, 'params') else 'N/A'}")
else:
    print("NOT FOUND in registry!")
    ids = sorted(all_methods.keys())
    print(f"  Available IDs (last 10): {ids[-10:]}")
