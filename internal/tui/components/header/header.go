package header

import (
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/alextheberge/opsy/internal/thememanager"
	"github.com/muesli/reflow/wrap"
)

// Model represents a header component that displays the current task.
type Model struct {
	task           string
	theme          thememanager.Theme
	containerStyle lipgloss.Style
	textStyle      lipgloss.Style
	maxWidth       int
}

// Option is a function that modifies the Model.
type Option func(*Model)

// New creates a new header Model with the given options.
// If no options are provided, it creates a header with default values.
func New(opts ...Option) *Model {
	m := &Model{
		task: "",
	}

	for _, opt := range opts {
		opt(m)
	}

	m.containerStyle = containerStyle(m.theme, m.maxWidth)
	m.textStyle = textStyle(m.theme)

	return m
}

// Init initializes the header Model.
// It implements the tea.Model interface.
func (m *Model) Init() tea.Cmd {
	return nil
}

// Update handles messages and updates the header Model accordingly.
// It implements the tea.Model interface.
func (m *Model) Update(msg tea.Msg) (*Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.maxWidth = msg.Width
		m.containerStyle = containerStyle(m.theme, m.maxWidth)
	}

	return m, nil
}

// View renders the header component.
func (m *Model) View() string {
	task := m.textStyle.Render(wrap.String(m.task, m.maxWidth-10))
	return m.containerStyle.Render(m.textStyle.Bold(true).Render("Task: ") + task)
}

// WithTask returns an Option that sets the task text in the header.
func WithTask(task string) Option {
	return func(m *Model) {
		m.task = task
	}
}

// WithTheme returns an Option that sets the theme for the header.
func WithTheme(theme thememanager.Theme) Option {
	return func(m *Model) {
		m.theme = theme
	}
}

// containerStyle creates a style for the container of the header component.
func containerStyle(theme thememanager.Theme, maxWidth int) lipgloss.Style {
	return lipgloss.NewStyle().
		Background(theme.BaseColors.Base01).
		Width(maxWidth).
		Padding(1, 2)
}

// textStyle creates a style for the text of the header component.
func textStyle(theme thememanager.Theme) lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(theme.BaseColors.Base04).
		Background(theme.BaseColors.Base01)
}
