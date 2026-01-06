import logging
import signal
import time


def main() -> None:
    """
    Minimal keep-alive loop so the container stays running.
    Replace this with your actual app entrypoint when ready.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stop = {"running": True}

    def _handle_exit(signum, frame):  # type: ignore[arg-type]
        logging.info("Received signal %s, shutting down.", signum)
        stop["running"] = False

    signal.signal(signal.SIGTERM, _handle_exit)
    signal.signal(signal.SIGINT, _handle_exit)

    logging.info("Container keep-alive started. Replace app/main.py with your real app.")
    while stop["running"]:
        time.sleep(1)
    logging.info("Container keep-alive exited.")


if __name__ == "__main__":
    main()
