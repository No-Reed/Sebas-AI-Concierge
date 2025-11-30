# 🎩 **Sebastian: The AI Digital Concierge**

> *"I am simply one hell of a digital butler."*

---

## 📖 Overview

Sebastian is a local, multi-agent desktop concierge designed to manage the "household chores" of a developer's digital life. While you focus on high-value creative work, Sebastian handles background maintenance, file organization, debugging, and session context management.

Submitted for the Google AI Agents Capstone (Concierge Agents Track).

---

## 💡 The Pitch

### The Problem

Developers lose hours weekly to **digital entropy**:

* Constant context switching
* Cluttered downloads/temp folders
* Micro-bugs that interrupt deep work

### The Solution

Sebastian functions as an **autonomous agentic OS overlay**:

* 👀 Monitors screen and file system
* 🛹 Sorts files automatically using `sorting_rules.json`
* 🛠️ Analyzes code, detects bugs, and provides fixes via hotkey or voice

### The Value

Sebastian reduces cognitive load and gives you a **maintenance-free digital environment**.

---

## ⚙️ Technical Architecture

Sebastian uses a **Hub-and-Spoke Multi-Agent Architecture** powered by Gemini 2.0 Flash.

### Core Components

#### 🧠 The Orchestrator

* Routes tasks
* Decides between parallel vs sequential agent execution
* Manages lifecycle and context

---

## 🕵️ Agents

* **CodeAnalyzer:** Parses visible code
* **BugDetector:** Finds syntax/logic errors
* **FixGenerator:** Generates line-level fixes
* **SecurityAuditor:** Checks for vulnerabilities
* **Memory Bank:** Persistent long-term memory
* **Watchdog:** Real-time file I/O watcher and sorter

---

## 🧰 Key Features

* ✅ Multi-agent architecture
* ✅ Tools Registry
* ✅ Context compaction
* ✅ Metrics and telemetry
* ✅ Independent background processes

---

## 📂 Project Structure

```plaintext
Sebastian/
│
├── Sebas.py                 # Main entry point, UI logic, and Agent definitions
│
├── sebastian_config.json    # Configuration for cooldowns, thresholds, and UI position
│
├── sorting_rules.json       # Taxonomy for the file organizer
│
├── memory_bank.json         # Persistent storage for agent memories
│
├── sessions.json            # Logs of interaction sessions and workflows
│
└── metrics.json             # Telemetry data for agent performance
```

---

## 🚀 Usage

**Start Sebastian**

```
python3 Sebas.py
```

**Trigger Hotkeys**

| Hotkey       | Action                                                                      |
| ------------ | --------------------------------------------------------------------------- |
| Ctrl+Shift+A | Auto-Analyze: Triggers the Multi-Agent workflow on the current screen/file. |
| Ctrl+Shift+S | Apply Fixes: Automatically applies generated code fixes to the active file. |
| Ctrl+Shift+M | Metrics: Exports usage data to metrics.json.                                |
| Ctrl+Shift+R | Restore: Brings the GUI back if minimized.                                  |

**Automatic File Sorting****
Files are categorized and moved based on the rules defined in `sorting_rules.json`.

---

## 🛡️ Security Notes

* Fully offline
* No uploading of user files or code
* Memory Bank stored as encrypted JSON

---

## 🧭 Roadmap

* GUI Dashboard
* Plugin Ecosystem
* AI-Assisted Refactoring Engine
* Multi-Device Sync
