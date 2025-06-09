package footer

import (
	"regexp"
	"sync"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/alextheberge/opsy/internal/agent"
	"github.com/alextheberge/opsy/internal/thememanager"
	"github.com/stretchr/testify/assert"
)

// stripANSI removes ANSI color codes from a string.
func stripANSI(str string) string {
	re := regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)
	return re.ReplaceAllString(str, "")
}

// TestNew tests the creation of a new footer component.
func TestNew(t *testing.T) {
	t.Run("creates with valid parameters", func(t *testing.T) {
		theme := thememanager.Theme{
			BaseColors: thememanager.BaseColors{
				Base01: "#000000",
				Base04: "#FFFFFF",
			},
		}
		params := Parameters{
			Engine:      "TestEngine",
			Model:       "TestModel",
			MaxTokens:   1000,
			Temperature: 0.7,
			ToolsCount:  5,
		}

		m := New(
			WithTheme(theme),
			WithParameters(params),
		)

		assert.NotNil(t, m)
		assert.Equal(t, params, m.parameters)
		assert.Equal(t, theme, m.theme)
		assert.Equal(t, agent.StatusReady, m.status)
	})

	t.Run("creates with nil theme", func(t *testing.T) {
		m := New()
		assert.NotNil(t, m)
		assert.Equal(t, thememanager.Theme{}, m.theme)
	})

	t.Run("creates with empty parameters", func(t *testing.T) {
		m := New(WithParameters(Parameters{}))
		assert.NotNil(t, m)
		assert.Equal(t, Parameters{}, m.parameters)
	})
}

// TestUpdate tests the update function of the footer component.
func TestUpdate(t *testing.T) {
	t.Run("handles window size message", func(t *testing.T) {
		theme := thememanager.Theme{
			BaseColors: thememanager.BaseColors{
				Base01: "#000000",
				Base04: "#FFFFFF",
			},
		}
		m := New(WithTheme(theme))

		newModel, cmd := m.Update(tea.WindowSizeMsg{Width: 100})
		assert.NotNil(t, newModel)
		assert.Nil(t, cmd)
		assert.Equal(t, 100, newModel.maxWidth)
	})

	t.Run("handles status update", func(t *testing.T) {
		m := New()
		newModel, cmd := m.Update(agent.Status("Running"))
		assert.NotNil(t, newModel)
		assert.Nil(t, cmd)
		assert.Equal(t, "Running", newModel.status)
	})
}

// TestView tests the view function of the footer component.
func TestView(t *testing.T) {
	t.Run("renders with all parameters", func(t *testing.T) {
		theme := thememanager.Theme{
			BaseColors: thememanager.BaseColors{
				Base01: "#000000",
				Base04: "#FFFFFF",
			},
		}
		params := Parameters{
			Engine:      "TestEngine",
			Model:       "TestModel",
			MaxTokens:   1000,
			Temperature: 0.7,
			ToolsCount:  5,
		}

		m := New(
			WithTheme(theme),
			WithParameters(params),
		)
		m.maxWidth = 100

		view := stripANSI(m.View())
		assert.Contains(t, view, "TestEngine")
		assert.Contains(t, view, "TestModel")
		assert.Contains(t, view, "1000")
		assert.Contains(t, view, "0.7")
		assert.Contains(t, view, "5")
		assert.Contains(t, view, "Ready")
	})

	t.Run("handles small window width", func(t *testing.T) {
		m := New(WithParameters(Parameters{
			Engine: "TestEngine",
			Model:  "TestModel",
		}))
		m.maxWidth = 40

		view := stripANSI(m.View())
		assert.NotEmpty(t, view)
		assert.Contains(t, view, "TestEngine")
	})

	t.Run("handles empty parameters", func(t *testing.T) {
		m := New()
		m.maxWidth = 100

		view := stripANSI(m.View())
		assert.NotEmpty(t, view)
		assert.Contains(t, view, "Ready")
	})
}

// TestInit tests the initialization of the footer component.
func TestInit(t *testing.T) {
	m := New()
	cmd := m.Init()
	assert.Nil(t, cmd)
}

// TestThemeChange tests the component's response to theme changes.
func TestThemeChange(t *testing.T) {
	initialTheme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#000000",
			Base04: "#FFFFFF",
		},
	}

	newTheme := thememanager.Theme{
		BaseColors: thememanager.BaseColors{
			Base01: "#111111",
			Base04: "#EEEEEE",
		},
	}

	params := Parameters{
		Engine:      "TestEngine",
		Model:       "TestModel",
		MaxTokens:   1000,
		Temperature: 0.7,
		ToolsCount:  5,
	}

	// Create and setup first model
	m1 := New(WithTheme(initialTheme), WithParameters(params))
	m1, _ = m1.Update(tea.WindowSizeMsg{Width: 100})

	// Create and setup second model
	m2 := New(WithTheme(newTheme), WithParameters(params))
	m2, _ = m2.Update(tea.WindowSizeMsg{Width: 100})

	// Verify container styles are different
	assert.NotEqual(t,
		m1.containerStyle.GetBackground(),
		m2.containerStyle.GetBackground(),
		"container styles should have different backgrounds",
	)

	// Verify text styles are different
	assert.NotEqual(t,
		m1.textStyle.GetForeground(),
		m2.textStyle.GetForeground(),
		"text styles should have different colors",
	)

	// Verify styles match their themes
	assert.Equal(t,
		lipgloss.Color(initialTheme.BaseColors.Base01),
		m1.containerStyle.GetBackground(),
		"container style should use Base01 color",
	)

	assert.Equal(t,
		lipgloss.Color(initialTheme.BaseColors.Base04),
		m1.textStyle.GetForeground(),
		"text style should use Base04 color",
	)

	assert.Equal(t,
		lipgloss.Color(newTheme.BaseColors.Base01),
		m2.containerStyle.GetBackground(),
		"container style should use Base01 color",
	)

	assert.Equal(t,
		lipgloss.Color(newTheme.BaseColors.Base04),
		m2.textStyle.GetForeground(),
		"text style should use Base04 color",
	)

	// Verify content is identical
	stripped1 := stripANSI(m1.View())
	stripped2 := stripANSI(m2.View())
	assert.Equal(t, stripped1, stripped2, "content should be same after stripping ANSI codes")
}

// TestConcurrentAccess tests thread safety of the footer component.
func TestConcurrentAccess(t *testing.T) {
	m := New()
	var wg sync.WaitGroup
	numGoroutines := 10

	// Test concurrent updates
	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			_, _ = m.Update(tea.WindowSizeMsg{Width: 100})
			_, _ = m.Update(agent.Status("Running"))
			_ = m.View()
		}()
	}
	wg.Wait()

	// Verify component is still in a valid state
	view := stripANSI(m.View())
	assert.NotEmpty(t, view)
	assert.Contains(t, view, "Running")
}
