import uvicorn


def main() -> None:
    uvicorn.run(
        "recalld.app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        access_log=False,
        # Keep shutdown nearly immediate, but allow active response tasks (e.g. SSE)
        # to wind down cleanly instead of being force-cancelled and logged as errors.
        timeout_graceful_shutdown=1,
    )


if __name__ == "__main__":
    main()
