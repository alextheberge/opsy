package main

import (
	"context"
	"errors"
	"log"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/alextheberge/opsy/internal/agent"
	"github.com/alextheberge/opsy/internal/config"
	"github.com/alextheberge/opsy/internal/thememanager"
	"github.com/alextheberge/opsy/internal/tool"
	"github.com/alextheberge/opsy/internal/toolmanager"
	"github.com/alextheberge/opsy/internal/tui"
)

const (
	// ErrNoTaskProvided is the error message for no task provided.
	ErrNoTaskProvided = "no task provided"
)

// main is the entry point for the Opsy application.
func main() {
	ctx := context.Background()

	task, err := getTask()
	if err != nil {
		log.Fatal(err)
	}

	cfg := config.New()
	if err := cfg.LoadConfig(); err != nil {
		log.Fatal(err)
	}

	logger, err := cfg.GetLogger()
	if err != nil {
		log.Fatal(err)
	}

	logger.With("task", task).Info("Started Opsy")

	themeManager := thememanager.New(thememanager.WithLogger(logger))
	if err := themeManager.LoadTheme(cfg.GetConfig().UI.Theme); err != nil {
		log.Fatal(err)
	}

	communication := &agent.Communication{
		Commands: make(chan tool.Command),
		Messages: make(chan agent.Message),
		Status:   make(chan agent.Status),
	}

	agnt := agent.New(
		agent.WithConfig(cfg.GetConfig()),
		agent.WithLogger(logger),
		agent.WithContext(ctx),
		agent.WithCommunication(communication),
	)

	toolManager := toolmanager.New(
		toolmanager.WithConfig(cfg.GetConfig()),
		toolmanager.WithLogger(logger),
		toolmanager.WithContext(ctx),
		toolmanager.WithAgent(agnt),
	)
	if err := toolManager.LoadTools(); err != nil {
		log.Fatal(err)
	}

	tui := tui.New(
		tui.WithTheme(themeManager.GetTheme()),
		tui.WithConfig(cfg.GetConfig()),
		tui.WithTask(task),
		tui.WithToolsCount(len(toolManager.GetTools())),
	)
	p := tea.NewProgram(tui, tea.WithAltScreen(), tea.WithMouseCellMotion(), tea.WithContext(ctx))

	go func() {
		if _, err := agnt.Run(&tool.RunOptions{Task: task, Tools: toolManager.GetTools()}, ctx); err != nil {
			communication.Status <- agent.StatusError
			logger.With("task", task).Error("Opsy finished with error", "error", err)
		} else {
			communication.Status <- agent.StatusFinished
			logger.With("task", task).Info("Opsy finished")
		}
	}()

	go func() {
		for msg := range communication.Messages {
			p.Send(msg)
		}
	}()

	go func() {
		for msg := range communication.Commands {
			p.Send(msg)
		}
	}()

	go func() {
		for msg := range communication.Status {
			p.Send(msg)
		}
	}()

	if _, err := p.Run(); err != nil {
		log.Fatal(err)
	}
}

// getTask returns the task from the command line arguments.
func getTask() (string, error) {
	if len(os.Args) > 1 && os.Args[1] != "" {
		return os.Args[1], nil
	}

	return "", errors.New(ErrNoTaskProvided)
}
