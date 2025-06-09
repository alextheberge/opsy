package tui

import (
	"math"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/alextheberge/opsy/internal/agent"
	"github.com/alextheberge/opsy/internal/config"
	"github.com/alextheberge/opsy/internal/thememanager"
	"github.com/alextheberge/opsy/internal/tool"
	"github.com/alextheberge/opsy/internal/tui/components/commandspane"
	"github.com/alextheberge/opsy/internal/tui/components/footer"
	"github.com/alextheberge/opsy/internal/tui/components/header"
	"github.com/alextheberge/opsy/internal/tui/components/messagespane"
)

// model is the main model for the TUI.
type model struct {
	theme        *thememanager.Theme
	header       *header.Model
	footer       *footer.Model
	messagesPane *messagespane.Model
	commandsPane *commandspane.Model
	config       config.Configuration
	task         string
	toolsCount   int
}

// Option is a function that configures the model.
type Option func(*model)

// New creates a new TUI instance.
func New(opts ...Option) *model {
	m := &model{
		config: config.New().GetConfig(),
		theme: &thememanager.Theme{
			BaseColors:   thememanager.BaseColors{},
			AccentColors: thememanager.AccentColors{},
		},
	}

	for _, opt := range opts {
		opt(m)
	}

	m.header = header.New(header.WithTheme(*m.theme), header.WithTask(m.task))
	m.footer = footer.New(footer.WithTheme(*m.theme), footer.WithParameters(footer.Parameters{
		Engine:      "Anthropic",
		Model:       m.config.Anthropic.Model,
		MaxTokens:   m.config.Anthropic.MaxTokens,
		Temperature: m.config.Anthropic.Temperature,
		ToolsCount:  m.toolsCount,
	}))
	m.messagesPane = messagespane.New(messagespane.WithTheme(*m.theme))
	m.commandsPane = commandspane.New(commandspane.WithTheme(*m.theme))

	return m
}

// Init initializes the TUI.
func (m *model) Init() tea.Cmd {
	return tea.SetWindowTitle("Opsy - Your AI-Powered SRE Colleague")
}

// Update handles all messages and updates the TUI
func (m *model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var headerCmd, footerCmd, messagesCmd, commandsCmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		if msg.String() == "ctrl+c" {
			return m, tea.Quit
		}
	case tea.WindowSizeMsg:
		headerHeight := int(math.Ceil(float64(lipgloss.Width(m.task))/float64(msg.Width))) * 2
		footerHeight := lipgloss.Height(m.footer.View())
		remainingHeight := msg.Height - headerHeight - footerHeight - 8

		m.header, headerCmd = m.header.Update(tea.WindowSizeMsg{
			Width:  msg.Width,
			Height: headerHeight,
		})
		m.footer, footerCmd = m.footer.Update(tea.WindowSizeMsg{
			Width:  msg.Width,
			Height: footerHeight,
		})
		m.messagesPane, messagesCmd = m.messagesPane.Update(tea.WindowSizeMsg{
			Width:  msg.Width,
			Height: remainingHeight * 2 / 3,
		})
		m.commandsPane, commandsCmd = m.commandsPane.Update(tea.WindowSizeMsg{
			Width:  msg.Width,
			Height: remainingHeight * 1 / 3,
		})
	case agent.Message:
		m.messagesPane, messagesCmd = m.messagesPane.Update(msg)
	case tool.Command:
		m.commandsPane, commandsCmd = m.commandsPane.Update(msg)
	default:
		m.header, headerCmd = m.header.Update(msg)
		m.footer, footerCmd = m.footer.Update(msg)
		m.messagesPane, messagesCmd = m.messagesPane.Update(msg)
		m.commandsPane, commandsCmd = m.commandsPane.Update(msg)
	}

	return m, tea.Batch(headerCmd, footerCmd, messagesCmd, commandsCmd)
}

// View renders the TUI.
func (m *model) View() string {
	return lipgloss.JoinVertical(lipgloss.Top,
		m.header.View(),
		m.messagesPane.View(),
		m.commandsPane.View(),
		m.footer.View(),
	)
}

// WithTask sets the task that the agent will execute.
func WithTask(task string) Option {
	return func(m *model) {
		m.task = task
	}
}

// WithConfig sets the configuration for the TUI.
func WithConfig(cfg config.Configuration) Option {
	return func(m *model) {
		m.config = cfg
	}
}

// WithTheme sets the theme for the TUI.
func WithTheme(theme *thememanager.Theme) Option {
	return func(m *model) {
		m.theme = theme
	}
}

// WithToolsCount sets the number of tools that the agent will use.
func WithToolsCount(toolsCount int) Option {
	return func(m *model) {
		m.toolsCount = toolsCount
	}
}
