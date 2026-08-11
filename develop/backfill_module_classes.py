"""Write full/module_classes.json for models traced before run.py started emitting it.

The map is `{module path: [source class names]}` and it is the join key between what the trace
records (module PATHS) and what the source declares (config reads per CLASS). Building it needs
the model, but only the model -- no trace, no weights, no inputs. A weightless meta construction
takes about a second even for a 550B checkpoint, so backfilling the whole fleet is cheap and does
not touch a single traced row.

Run:   .venv\\Scripts\\python.exe develop\\backfill_module_classes.py [substring-filter]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import yaml

import introspect
import loader
import provenance

MODELS = os.path.join(os.path.dirname(__file__), "models")
OUT = os.path.join(os.path.dirname(__file__), "..", "models")


def main(filt=None):
    done = skipped = failed = 0
    for prof_path in sorted(os.listdir(MODELS)):
        if not prof_path.endswith(".yaml"):
            continue
        prof = yaml.safe_load(open(os.path.join(MODELS, prof_path), encoding="utf-8"))
        mid = prof["model_id"]
        if filt and filt not in mid and filt not in prof_path:
            continue
        d = os.path.join(OUT, mid.replace("/", "__"), "full")
        if not os.path.isdir(d):
            continue
        out = os.path.join(d, "module_classes.json")
        if os.path.exists(out):
            skipped += 1
            continue
        t0 = time.time()
        try:
            cfg, prov = provenance.snapshot(mid, prof.get("revision"),
                                            config_overrides=prof.get("config_overrides"))
            model = loader.load_meta(cfg, trust_remote_code=prov["trust_remote_code"])
            classes = introspect.module_classes(model)
        except Exception as e:  # a model we cannot build yields no map -- and says so, rather
            # than leaving an empty file that would read as "this module reads nothing".
            print("FAIL %-52s %s: %s" % (mid, type(e).__name__, str(e)[:120]))
            failed += 1
            continue
        with open(out, "w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False, indent=1)
        print("ok   %-52s %4d modules  %5.1fs" % (mid, len(classes), time.time() - t0))
        done += 1
    print("\n작성 %d / 이미 있음 %d / 실패 %d" % (done, skipped, failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
