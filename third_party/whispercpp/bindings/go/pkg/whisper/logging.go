package whisper

import base "github.com/ggerganov/whisper.cpp/bindings/go"

// SetLogSink forwards whisper.cpp log messages to the provided callback.
func SetLogSink(fn func(level int, text string)) (prev func(level int, text string)) {
	return base.SetLogSink(fn)
}
