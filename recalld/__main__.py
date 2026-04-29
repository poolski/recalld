import uvicorn


def main() -> None:
    uvicorn.run(
        "recalld.app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
