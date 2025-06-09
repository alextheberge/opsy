package commandspane

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/alextheberge/opsy/internal/thememanager"
	"github.com/alextheberge/opsy/internal/tool"
	"github.com/muesli/reflow/wrap"
)

// Model represents the commands pane component.
// It maintains the state of the commands list and viewport,
// handling command history and display formatting.
type Model struct {
	// theme defines the color scheme for the component
	theme thememanager.Theme
	// maxWidth is the maximum width of the component
	maxWidth int
	// maxHeight is the maximum height of the component
	maxHeight int
	// viewport handles scrollable content display
	viewport viewport.Model
	// commands stores the history of executed commands
	commands []tool.Command
}

// Option is a function that modifies the Model.
type Option func(*Model)

// title is the title of the commands pane.
const title = "Commands"

// New creates a new commands pane component.
func New(opts ...Option) *Model {
	m := &Model{
		viewport: viewport.New(0, 0),
		commands: []tool.Command{},
	}

	for _, opt := range opts {
		opt(m)
	}

	return m
}

// Init initializes the commands pane component.
func (m *Model) Init() tea.Cmd {
	return nil
}

// Update handles messages and updates the commands pane component.
func (m *Model) Update(msg tea.Msg) (*Model, tea.Cmd) {
	var cmd tea.Cmd
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.maxWidth = msg.Width - 6
		m.maxHeight = msg.Height
		m.viewport.Width = m.maxWidth
		m.viewport.Height = msg.Height
		m.viewport.Style = lipgloss.NewStyle().Background(m.theme.BaseColors.Base01)

		// Rerender all commands with new dimensions
		if len(m.commands) > 0 {
			m.renderCommands()
		} else {
			m.viewport.SetContent(m.titleStyle().Render(title))
		}
	case tool.Command:
		m.commands = append(m.commands, msg)
		m.renderCommands()
		m.viewport.GotoBottom()
	}

	m.viewport, cmd = m.viewport.Update(msg)
	return m, cmd
}

// View renders the commands pane component.
func (m *Model) View() string {
	return m.containerStyle().Render(m.viewport.View())
}

// WithTheme sets the theme for the commands pane component.
func WithTheme(theme thememanager.Theme) Option {
	return func(m *Model) {
		m.theme = theme
	}
}

// containerStyle creates a style for the container of the commands pane component.
func (m *Model) containerStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Background(m.theme.BaseColors.Base01).
		Padding(1, 2).
		Border(lipgloss.NormalBorder(), true).
		BorderForeground(m.theme.BaseColors.Base02).
		BorderBackground(m.theme.BaseColors.Base00)
}

// commandStyle creates a style for the command text.
func (m *Model) commandStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(m.theme.AccentColors.Accent0).
		Background(m.theme.BaseColors.Base01)
}

// timestampStyle creates a style for the timestamp of the commands pane component.
func (m *Model) timestampStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(m.theme.BaseColors.Base03).
		Background(m.theme.BaseColors.Base01).
		PaddingRight(1)
}

// workdirStyle creates a style for the working directory.
func (m *Model) workdirStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(m.theme.BaseColors.Base04).
		Background(m.theme.BaseColors.Base03).
		Margin(0, 1, 0, 0).
		MarginBackground(m.theme.BaseColors.Base01).
		Padding(0, 1)
}

// titleStyle creates a style for the title.
func (m *Model) titleStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(m.theme.BaseColors.Base04).
		Background(m.theme.BaseColors.Base01).
		Bold(true).
		Width(m.maxWidth)
}

// renderCommands formats and renders all commands
func (m *Model) renderCommands() {
	output := strings.Builder{}
	content := strings.Builder{}
	content.WriteString(m.titleStyle().Render(title))
	content.WriteString("\n\n")

	for _, cmd := range m.commands {
		timestamp := m.timestampStyle().Render(fmt.Sprintf("[%s]", cmd.StartedAt.Format("15:04:05")))
		workdir := m.workdirStyle().Render(cmd.WorkingDirectory)

		// Calculate available width for command
		commandWidth := m.maxWidth - lipgloss.Width(timestamp) - lipgloss.Width(workdir)

		// Always wrap the command to ensure consistent formatting
		wrappedCommand := wrap.String(cmd.Command, commandWidth)

		// Split wrapped command into lines
		commandLines := strings.Split(wrappedCommand, "\n")

		// Render first line with timestamp and workdir
		firstLine := m.commandStyle().Width(commandWidth).Render(commandLines[0])
		content.WriteString(fmt.Sprintf("%s%s%s", timestamp, workdir, firstLine))
		content.WriteString("\n")

		// Render remaining lines with proper indentation
		if len(commandLines) > 1 {
			indent := strings.Repeat(" ", lipgloss.Width(timestamp)+lipgloss.Width(workdir))
			for _, line := range commandLines[1:] {
				content.WriteString(indent)
				content.WriteString(m.commandStyle().Width(commandWidth).Render(line))
				content.WriteString("\n")
			}
		}
		content.WriteString("\n")
	}

	// Wrap all content in a background-styled container
	contentStyle := lipgloss.NewStyle().
		Background(m.theme.BaseColors.Base01).
		Width(m.maxWidth).
		Height(m.maxHeight)

	output.WriteString(contentStyle.Render(content.String()))
	m.viewport.SetContent(output.String())
}
