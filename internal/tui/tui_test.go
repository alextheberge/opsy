package tui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/alextheberge/opsy/internal/config"
	"github.com/alextheberge/opsy/internal/thememanager"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNew tests the creation of a new TUI model with various options.
func TestNew(t *testing.T) {
	t.Run("default configuration", func(t *testing.T) {
		m := New()
		require.NotNil(t, m)
		assert.NotNil(t, m.theme)
		assert.NotNil(t, m.header)
		assert.NotNil(t, m.footer)
		assert.NotNil(t, m.messagesPane)
		assert.NotNil(t, m.commandsPane)
	})

	t.Run("with custom options", func(t *testing.T) {
		cfg := config.Configuration{
			Anthropic: config.AnthropicConfiguration{
				Model:       "test-model",
				MaxTokens:   1000,
				Temperature: 0.7,
			},
		}
		theme := &thememanager.Theme{
			BaseColors:   thememanager.BaseColors{},
			AccentColors: thememanager.AccentColors{},
		}
		task := "test task"
		toolsCount := 5

		m := New(
			WithConfig(cfg),
			WithTheme(theme),
			WithTask(task),
			WithToolsCount(toolsCount),
		)

		require.NotNil(t, m)
		assert.Equal(t, cfg, m.config)
		assert.Equal(t, theme, m.theme)
		assert.Equal(t, task, m.task)
		assert.Equal(t, toolsCount, m.toolsCount)
	})
}

// TestModel_Init tests the initialization of the TUI model.
func TestModel_Init(t *testing.T) {
	m := New()
	cmd := m.Init()
	require.NotNil(t, cmd)
}

// TestModel_Update tests the update function of the TUI model.
func TestModel_Update(t *testing.T) {
	t.Run("quit on ctrl+c", func(t *testing.T) {
		m := New()
		model, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
		assert.NotNil(t, model)
		assert.NotNil(t, cmd)
	})

	t.Run("handle window size message", func(t *testing.T) {
		m := New()
		updatedModel, _ := m.Update(tea.WindowSizeMsg{
			Width:  100,
			Height: 50,
		})
		assert.NotNil(t, updatedModel)

		// Verify that the message was processed by checking if components exist
		tuiModel, ok := updatedModel.(*model)
		assert.True(t, ok, "expected model to be of type *model")
		assert.NotNil(t, tuiModel.header)
		assert.NotNil(t, tuiModel.footer)
		assert.NotNil(t, tuiModel.messagesPane)
		assert.NotNil(t, tuiModel.commandsPane)
	})
}

// TestModel_View tests the view rendering of the TUI model.
func TestModel_View(t *testing.T) {
	m := New()
	view := m.View()
	assert.NotEmpty(t, view)
}
