package messagespane

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/alextheberge/opsy/internal/agent"
	"github.com/alextheberge/opsy/internal/thememanager"
)

// Model represents the messages pane component.
type Model struct {
	theme     thememanager.Theme
	maxWidth  int
	maxHeight int
	viewport  viewport.Model
	messages  []agent.Message
}

// Option is a function that modifies the Model.
type Option func(*Model)

// New creates a new messages pane component.
func New(opts ...Option) *Model {
	m := &Model{
		viewport: viewport.New(0, 0),
		messages: []agent.Message{},
	}

	for _, opt := range opts {
		opt(m)
	}

	return m
}

// title is the title of the messages pane.
const title = "Messages"

// Init initializes the messages pane component.
func (m *Model) Init() tea.Cmd {
	return nil
}

// Update handles messages and updates the messages pane component.
func (m *Model) Update(msg tea.Msg) (*Model, tea.Cmd) {
	var cmd tea.Cmd
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.maxWidth = msg.Width - 6
		m.maxHeight = msg.Height
		m.viewport.Width = m.maxWidth
		m.viewport.Height = msg.Height
		m.viewport.Style = lipgloss.NewStyle().Background(m.theme.BaseColors.Base01)

		// Rerender all messages with new dimensions
		if len(m.messages) > 0 {
			m.renderMessages()
		} else {
			m.viewport.SetContent(m.titleStyle().Render(title))
		}
	case agent.Message:
		m.messages = append(m.messages, msg)
		m.renderMessages()
		m.viewport.GotoBottom()
	}

	m.viewport, cmd = m.viewport.Update(msg)
	return m, cmd
}

// View renders the messages pane component.
func (m *Model) View() string {
	return m.containerStyle().Render(m.viewport.View())
}

// WithTheme sets the theme for the messages pane component.
func WithTheme(theme thememanager.Theme) Option {
	return func(m *Model) {
		m.theme = theme
	}
}

// containerStyle creates a style for the container of the messages pane component.
func (m *Model) containerStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Background(m.theme.BaseColors.Base01).
		Padding(1, 2).
		Border(lipgloss.NormalBorder(), true).
		BorderForeground(m.theme.BaseColors.Base02).
		BorderBackground(m.theme.BaseColors.Base00).
		UnsetBorderBottom()
}

// messageStyle creates a style for the text of the messages pane component.
func (m *Model) messageStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(m.theme.BaseColors.Base04).
		Background(m.theme.BaseColors.Base03).
		Margin(1, 0, 1, 0).
		Padding(1, 2, 1, 1).
		MarginBackground(m.theme.BaseColors.Base01).
		Width(m.maxWidth)
}

// timestampStyle creates a style for the timestamp of the messages pane component.
func (m *Model) timestampStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(m.theme.BaseColors.Base03).
		Background(m.theme.BaseColors.Base01).
		PaddingRight(1)
}

// authorStyle creates a style for author messages.
func (m *Model) authorStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(m.theme.AccentColors.Accent1).
		Background(m.theme.BaseColors.Base01).
		Width(m.maxWidth).
		Bold(true)
}

// titleStyle creates a style for the title.
func (m *Model) titleStyle() lipgloss.Style {
	return lipgloss.NewStyle().
		Foreground(m.theme.BaseColors.Base04).
		Background(m.theme.BaseColors.Base01).
		Bold(true).
		Width(m.maxWidth)
}

// renderMessages formats and renders all messages
func (m *Model) renderMessages() {
	output := strings.Builder{}
	output.WriteString(m.titleStyle().Render(title))
	output.WriteString("\n\n")

	for _, message := range m.messages {
		timestamp := m.timestampStyle().Render(fmt.Sprintf("[%s]", message.Timestamp.Format("15:04:05")))
		authorStyle := m.authorStyle().Width(m.maxWidth - lipgloss.Width(timestamp))
		author := agent.Name

		if message.Tool != "" {
			author = fmt.Sprintf("%s->%s", agent.Name, message.Tool)
			authorStyle = authorStyle.Foreground(m.theme.AccentColors.Accent2)
		}

		author = authorStyle.Render(fmt.Sprintf("%s:", author))
		messageText := m.messageStyle().Render(sanitizeMessage(message.Message))

		output.WriteString(fmt.Sprintf("%s%s", timestamp, author))
		output.WriteString("\n")
		output.WriteString(messageText)
		output.WriteString("\n")
	}

	m.viewport.SetContent(output.String())
}

// sanitizeMessage removes unnecessary symbols from the message.
func sanitizeMessage(message string) string {
	// Remove XML-style tags from the message
	message = regexp.MustCompile(`<[^>]*>`).ReplaceAllString(message, "")

	// Remove trailing spaces and newlines
	message = strings.TrimSpace(message)
	message = strings.Trim(message, "\n")

	return message
}
