.PHONY: setup build run test lint fmt clean whispercpp

setup:
	go mod download

whispercpp:
	cmake -S third_party/whispercpp -B third_party/whispercpp/build -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_SERVER=OFF -DGGML_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release
	cmake --build third_party/whispercpp/build --target whisper -j4

build: whispercpp
	go build -o bin/recalld .

run: whispercpp
	go run .

test: whispercpp
	go test ./...

lint: whispercpp
	go vet ./...

fmt:
	gofmt -w *.go

clean:
	rm -rf bin third_party/whispercpp/build .pytest_cache __pycache__ */__pycache__
