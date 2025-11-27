# OpenAI Model for ADK-Go

This package provides an OpenAI model implementation for the ADK-Go framework.

## Usage
`go get github.com/byebyebruce/adk-go-openai`


```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	openai "github.com/byebyebruce/adk-go-openai"
	go_openai "github.com/sashabaranov/go-openai"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type Input struct {
	City string `json:"city" jsonschema:"city name"`
}

type Output struct {
	WeatherSummary string `json:"weather_summary" jsonschema:"weather summary in the given city"`
}

func GetWeather(ctx tool.Context, input Input) (Output, error) {
	log.Println("Getting weather for city: ", input.City)
	return Output{
		WeatherSummary: fmt.Sprintf("Today in %q is sunny\n", input.City),
	}, nil
}

func main() {
	ctx := context.Background()

	modelName := "gpt-5.1"
	openaiCfg := go_openai.DefaultConfig(os.Getenv("OPENAI_API_KEY"))
	if baseURL := os.Getenv("OPENAI_BASE_URL"); baseURL != "" {
		openaiCfg.BaseURL = baseURL
	}

	weatherTool, err := functiontool.New(functiontool.Config{
		Name:        "weather",
		Description: "Returns weather in a city",
	}, GetWeather)
	if err != nil {
		log.Fatalf("Failed to create tool: %v", err)
	}

	model := openai.NewOpenAIModel(modelName, openaiCfg)
	a, err := llmagent.New(llmagent.Config{
		Name:        "weather_agent",
		Model:       model,
		Description: "Agent to answer questions about weather in a city.",
		Instruction: "Your SOLE purpose is to answer questions about the current weather in a specific city. You MUST refuse to answer any questions unrelated to weather.",
		Tools: []tool.Tool{
			weatherTool,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
```
## Example
[Example](example)