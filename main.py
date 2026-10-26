import argparse
import sys
import time


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="FightWatch — Fight-only Video Detection"
    )
    subcommands = parser.add_subparsers(dest="mode", required=True)
    serve = subcommands.add_parser("serve", help="Video and iPhone camera dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    run = subcommands.add_parser(
        "run", help="Analyze a video, camera index, or stream URL"
    )
    run.add_argument("--source", required=True)
    run.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args()
    if args.mode == "serve":
        import uvicorn

        uvicorn.run("api.fight_server:app", host=args.host, port=args.port)
        return
    from api.fight_server import FightProcessor

    processor = FightProcessor()
    source = int(args.source) if args.source.isdigit() else args.source
    processor.start(source, args.allow_pilot)
    try:
        while processor.is_running():
            print(processor.status(), flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        processor.stop()
    result = processor.status()
    print(result)
    if result["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
