# 🎩 Sebastian: The AI Digital Concierge

"I am simply one hell of a digital butler."

---

## 📖 Overview

Sebastian is a local, multi-agent desktop concierge designed to manage the "household chores" of a developer's digital life. While you focus on high-value creative work, Sebastian handles background maintenance, file organization, debugging, and session context management.

Submitted for the Google AI Agents Capstone (Freestyle / Concierge Agents Track).

---

## 💡 The Pitch

### The Problem

Developers lose hours weekly to digital entropy:

- **Context Switching:** Breaking flow to copy-paste errors into a browser AI.
- **File Clutter:** Downloads and temp folders becoming unmanageable dumps.
- **Micro-Friction:** Repetitive tasks like formatting JSON or writing boilerplate emails.

### The Solution

Sebastian functions as an autonomous agentic OS overlay that lives on your machine:

- 👀 **Vision-Based:** Monitors your screen pixel-by-pixel to detect when you are stuck.
- 🛹 **Auto-Sorting:** Automatically organizes files into semantic categories (Finance, Dev, Personal) using `sorting_rules.json`.
- 🛠️ **Proactive Help:** Analyzes code, detects bugs, and provides fixes via a non-intrusive HUD.

### The Value

Sebastian reduces cognitive load and gives you a maintenance-free digital environment, saving **5–10 hours** of "digital housekeeping" per week.

---

## ⚙️ Technical Architecture

Sebastian uses a **Local Loop-based Agentic Architecture** powered by the multimodal capabilities of **Google Gemini 2.0 Flash**.

### Core Components

#### 🧠 The Brain (Gemini 2.0)

- Uses Gemini's multimodal vision capabilities to "see" screen state directly.
- Dynamic system prompts inject task-specific personas.

#### 👁️ Perception Layer

- **Vision:** `pyautogui` captures screen state; `ImageChops` detects motion/idleness.
- **File System:** `watchdog` monitors OS events (Created, Moved, Modified) in real-time.

#### 💾 Memory Layer

- **Short-Term:** Thread-safe real-time state for HUD updates.
- **Long-Term:** JSON-based persistence (`sebas_long_term_memory.json`).

#### 🦾 Action Layer

- **HUD:** Custom tkinter overlay offers contextual assistance.
- **File Ops:** Autonomous file moving, renaming, and content generation.

---

## 🧰 Key Features (Capstone Checklist)

- ✅ Multimodality (Screen Images + Text)
- ✅ Custom Tools for code analysis & file manipulation
- ✅ Long-Term Memory with MemoryManager
- ✅ Observability via `sebas.log`

---

## 📂 Project Structure

```
Sebastian/
│
├── Sebas.py                     # Main entry point, UI logic, and Agent definitions
│
├── sebas_config.json            # Configuration for cooldowns, thresholds, and UI position
│
├── sorting_rules.json           # Taxonomy for the file organizer (User Customizable)
│
├── sebas_long_term_memory.json  # Persistent brain for agent memories
│
└── sebas.log                    # Telemetry and event logs
```

---

## 🚀 How to Run (Local Deployment)

Sebastian is designed as a **Local Desktop Agent** for privacy and OS-level control.

### 1. Clone the Repo

```
git clone https://github.com/No-Reed/Sebas-AI-Concierge.git
cd Sebas-AI-Concierge
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Setup Keys

Create a `.env` file:

```
GEMINI_API_KEY=...
```

### 4. Launch Sebastian

```
python Sebas.py
```

### 5. Interact

- **Auto-Mode:** Simply work; Sebastian analyzes when you're idle or stuck.
- **Manual:** Use the HUD buttons to "Analyze" or "Chat".

---
## .exe Deployment Requirement -
---
### 1. Download the sorting_rules.json, .env(enter your own API inside it), an dthe Sebas.exe 
---
### 2. Put them inside same directory
---
### 3. Run the Sebas.exe file
---
## ⌨️ Hotkeys

| Hotkey       | Action                                                                      |
| ------------ | --------------------------------------------------------------------------- |
| Ctrl+Shift+A | Auto-Analyze: Triggers the Multi-Agent workflow on the current screen/file. |
| Ctrl+Shift+S | Apply Fixes: Automatically applies generated code fixes.                    |
| Ctrl+Shift+M | Metrics: Exports usage data to `metrics.json`.                              |
| Ctrl+Shift+R | Restore: Reopens the GUI if minimized.                                      |

---

## 🛡️ Security & Privacy

- **Local-First:** No files are uploaded except specific snippets for Gemini API calls.
- **Encrypted Keys:** API tokens are stored in `.env`.
- **No Cloud Storage:** All memory and logs are stored locally.

---

## 🧭 Roadmap

- [ ] GUI Dashboard for configuring sorting rules.
- [ ] Voice Interaction mode.
- [ ] Multi-Device Sync for memory sharing.

