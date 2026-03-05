# SWE Agent 分析

## 嚴格序列狀態機 (Strict Sequential State Machine, SSSM)

我在使用 Antigravity, Claude code, OpenCode 時，發現會有兩種任務追蹤狀態：

### Method 1. Todo List Markdown

Antigravity 在創建Todo list 時只是單純的markdown

### Method 2. Memory-based State Machine

Claude code 與 OpenCode (Opencode 只有在 build 模式下會存在，而且其不一定會選用該tool) 在創建Todo list 時會建立一個Memory-based State Machine
