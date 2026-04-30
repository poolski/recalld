module github.com/poolski/recalld

go 1.23

toolchain go1.24.12

require github.com/spf13/cobra v1.8.1

require github.com/alphacep/vosk-api/go v0.3.50 // indirect

require (
	github.com/ggerganov/whisper.cpp/bindings/go v0.0.0-20260420051257-fc674574ca27
	github.com/google/go-cmp v0.5.9 // indirect
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/spf13/pflag v1.0.5 // indirect
	gotest.tools/v3 v3.5.2 // indirect
)

replace github.com/ggerganov/whisper.cpp/bindings/go => ./third_party/whispercpp/bindings/go
