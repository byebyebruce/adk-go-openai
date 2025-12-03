package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	openai "github.com/byebyebruce/adk-go-openai"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	go_openai "github.com/sashabaranov/go-openai"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/mcptoolset"
	"google.golang.org/genai"
)

var (
	deepWikiMCPFlag = flag.String("deepwiki-mcp", "https://mcp.deepwiki.com/mcp", "DeepWiki MCP URL")
	modelName       = flag.String("model", "gpt-5.1", "OpenAI model name, default is gpt-5.1")
)

func main() {
	flag.Parse()

	mcpTransport := &mcp.StreamableClientTransport{
		Endpoint: *deepWikiMCPFlag,
	}
	mcpToolSet, err := mcptoolset.New(mcptoolset.Config{
		Transport: mcpTransport,
	})
	if err != nil {
		log.Fatalf("Failed to create MCP tool set: %v", err)
	}
	openaiCfg := go_openai.DefaultConfig(os.Getenv("OPENAI_API_KEY"))
	if baseURL := os.Getenv("OPENAI_BASE_URL"); baseURL != "" {
		openaiCfg.BaseURL = baseURL
	}
	model := openai.NewOpenAIModel(*modelName, openaiCfg)
	a, err := llmagent.New(llmagent.Config{
		Name:        "deepwiki_agent",
		Model:       model,
		Description: "Agent to answer questions about DeepWiki.",
		Instruction: "Your SOLE purpose is to answer questions about Github Repos.",
		Toolsets: []tool.Toolset{
			mcpToolSet,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	ctx := context.Background()
	sessionService := session.InMemoryService()
	sessionID := ""

	runner, err := runner.New(runner.Config{
		Agent:           a,
		AppName:         "test_app",
		SessionService:  sessionService,
		ArtifactService: artifact.InMemoryService(),
		MemoryService:   memory.InMemoryService(),
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		if sessionID == "" {
			resp, err := sessionService.Create(ctx, &session.CreateRequest{
				AppName: "test_app",
				UserID:  "test_user",
			})
			if err != nil {
				log.Fatalf("Failed to create session: %v", err)
			}
			sessionID = resp.Session.ID()
			fmt.Println("Session created: ", sessionID)
		}

		fmt.Println()
		fmt.Print("\nUser -> ")
		userInput, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		userInput = strings.TrimSpace(userInput)
		if userInput == `/exit` {
			return
		}
		if userInput == `/clear` {
			sessionID = ""
			continue
		}
		userMsg := genai.NewContentFromText(userInput, genai.RoleUser)
		seq := runner.Run(ctx, "test_user", sessionID, userMsg, agent.RunConfig{
			StreamingMode: agent.StreamingModeSSE,
		})
		fmt.Println("\nAgent -> ")
		for event, err := range seq {
			if err != nil {
				fmt.Printf("\nAGENT_ERROR: %v\n", err)
				break
			}
			if event.LLMResponse.Content == nil {
				continue
			}
			text := ""
			for _, part := range event.LLMResponse.Content.Parts {
				if part.FunctionCall != nil {
					fmt.Println("Function call: ", part.FunctionCall.Name, part.FunctionCall.Args)
				}
				if part.FunctionResponse != nil {
					fmt.Println("Function response: ", part.FunctionResponse.Name, part.FunctionResponse.Response)
				}
				text += part.Text
			}
			fmt.Print(text)
		}
	}
}
