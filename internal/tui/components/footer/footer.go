package footer

import (
	"strconv"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/alextheberge/opsy/internal/agent"
	"github.com/alextheberge/opsy/internal/thememanager"
)

// Model represents the footer component.
type Model struct {
	theme          thememanager.Theme
	parameters     Parameters
	containerStyle lipgloss.Style
	textStyle      lipgloss.Style
	maxWidth       int
	status         string
}

// Parameters represent the parameters of the application.
type Parameters struct {
	Engine      string
	Model       string
	MaxTokens   int64
	Temperature float64
	ToolsCount  int
}

// Option is a function that modifies the Model.
type Option func(*Model)

// New creates a new footer component.
func New(opts ...Option) *Model {
	m := &Model{
		status:     agent.StatusReady,
		parameters: Parameters{},
	}

	for _, opt := range opts {
		opt(m)
	}

	m.containerStyle = containerStyle(m.theme, m.maxWidth)
	m.textStyle = textStyle(m.theme)

	return m
}

// Init initializes the footer component.
func (m *Model) Init() tea.Cmd {
	return nil
}

// Update handles messages and updates the footer component.
func (m *Model) Update(msg tea.Msg) (*Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.maxWidth = msg.Width
		m.containerStyle = containerStyle(m.theme, m.maxWidth)
	case agent.Status:
		m.status = string(msg)
	}

	return m, nil
}

// View renders the footer component.
func (m *Model) View() string {
	footer := m.textStyle.Bold(true).Render("Engine: ") + m.textStyle.Render(m.parameters.Engine)
	footer += m.textStyle.Render(" | ") + m.textStyle.Bold(true).Render("Model: ") + m.textStyle.Render(m.parameters.Model)
	footer += m.textStyle.Render(" | ") + m.textStyle.Bold(true).Render("Temperature: ") + m.textStyle.Render(strconv.FormatFloat(m.parameters.Temperature, 'f', -1, 64))
	footer += m.textStyle.Render(" | ") + m.textStyle.Bold(true).Render("Max Tokens: ") + m.textStyle.Render(strconv.FormatInt(m.parameters.MaxTokens, 10))
	footer += m.textStyle.Render(" | ") + m.textStyle.Bold(true).Render("Tools: ") + m.textStyle.Render(strconv.Itoa(m.parameters.ToolsCount))

	footerStatus := m.textStyle.Bold(true).Render("Status: ") + m.textStyle.Render(m.status)
	footer += m.textStyle.Width(m.maxWidth - lipgloss.Width(footer) - 4).Align(lipgloss.Right).Render(footerStatus)

	return m.containerStyle.Render(footer)
}

// WithTheme sets the theme for the footer component.
func WithTheme(theme thememanager.Theme) Option {
	return func(m *Model) {
		m.theme = theme
		m.containerStyle = containerStyle(theme, m.maxWidth)
		m.textStyle = textStyle(theme)
	}
}

// WithParameters sets the parameters for the footer component.
func WithParameters(parameters Parameters) Option {
	return func(m *Model) {
		m.parameters = parameters
	}
}

// containerStyle creates a style for the container of the footer component.
func containerStyle(theme thememanager.Theme, maxWidth int) lipgloss.Style {
	return lipgloss.NewStyle().
		Background(theme.BaseColors.Base01).
		Width(maxWidth).
		Padding(1, 2, 1, 2)
}

// textStyle creates a style for the text of the footer component.
func textStyle(theme thememanager.Theme) lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(theme.BaseColors.Base04).
		Background(theme.BaseColors.Base01)
}
