package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

type options struct {
	addr        string
	configPath  string
	scratchRoot string
}

func defaultOptions() options {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return options{
		addr:        envOr("ADDR", "127.0.0.1:8765"),
		configPath:  filepath.Join(home, ".config", "recalld", "config.json"),
		scratchRoot: filepath.Join(home, ".local", "share", "recalld", "jobs"),
	}
}

func Execute() error {
	return newRootCommand().Execute()
}

func newRootCommand() *cobra.Command {
	opts := defaultOptions()

	cmd := &cobra.Command{
		Use:   "recalld",
		Short: "Local meeting note automation",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runServer(opts)
		},
		SilenceUsage: true,
	}

	cmd.Flags().StringVar(&opts.addr, "addr", opts.addr, "HTTP listen address")
	cmd.Flags().StringVar(&opts.configPath, "config", opts.configPath, "Config file path")
	cmd.Flags().StringVar(&opts.scratchRoot, "scratch", opts.scratchRoot, "Scratch directory")
	return cmd
}

func runServer(opts options) error {
	application, err := NewApp(opts.configPath, opts.scratchRoot, filepath.Join("recalld", "static"))
	if err != nil {
		return err
	}

	srv := &http.Server{
		Addr:    opts.addr,
		Handler: application.Handler(),
	}

	fmt.Printf("recalld listening on %s\n", opts.addr)
	return srv.ListenAndServe()
}

func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
