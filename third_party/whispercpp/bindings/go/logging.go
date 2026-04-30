package whisper

/*
#cgo CFLAGS: -I${SRCDIR}/../../include -I${SRCDIR}/../../ggml/include
#include <whisper.h>

extern void goWhisperLogCallback(int level, char * text);

static void whisper_log_cb(enum ggml_log_level level, const char * text, void * user_data) {
	(void) user_data;
	goWhisperLogCallback((int) level, (char *) text);
}

static void whisper_log_install(void) {
	whisper_log_set(whisper_log_cb, NULL);
}
*/
import "C"

import (
	"sync"
)

var (
	logSinkMu sync.RWMutex
	logSink   func(level int, text string)
)

func init() {
	C.whisper_log_install()
}

// SetLogSink registers a callback for whisper.cpp log messages.
// Passing nil disables forwarding to the Go callback.
func SetLogSink(fn func(level int, text string)) (prev func(level int, text string)) {
	logSinkMu.Lock()
	prev = logSink
	logSink = fn
	logSinkMu.Unlock()
	return prev
}

//export goWhisperLogCallback
func goWhisperLogCallback(level C.int, text *C.char) {
	logSinkMu.RLock()
	sink := logSink
	logSinkMu.RUnlock()
	if sink == nil {
		return
	}
	sink(int(level), C.GoString(text))
}
