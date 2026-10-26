import sys

from core.config import settings
from fight.runtime import FightRuntime


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    runtime = FightRuntime(settings.models_dir, allow_pilot=True)
    print(runtime.message)
    if runtime.model_ready:
        print(f"Architecture: {runtime.metadata.get('architecture')}")
        print(f"Approved for demo: {runtime.metadata.get('approved_for_demo', False)}")
    elif (settings.models_dir / "fight_model.ts").exists():
        raise SystemExit("Invalid fight runtime bundle")


if __name__ == "__main__":
    main()
