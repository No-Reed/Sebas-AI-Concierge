import os
import time
import threading
import logging
import json
import hashlib
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import tkinter as tk
from tkinter import font, scrolledtext, messagebox, filedialog
import pyautogui
import keyboard
from PIL import Image, ImageChops
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import google.generativeai as genai
from dotenv import load_dotenv
import re
from collections import deque
from PIL import Image, ImageChops, ImageStat
import uuid
import speech_recognition as sr
import pyttsx3

load_dotenv()

# LOGGING & OBSERVABILITY

class MetricsCollector:
    """Centralized metrics collection for observability"""
    def __init__(self):
        self.metrics = {
            "agent_calls": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time": [],
            "tool_usage": {},
            "agent_performance": {}
        }
        self.lock = threading.Lock()
    
    def record_metric(self, metric_name: str, value: Any):
        with self.lock:
            if metric_name in self.metrics:
                if isinstance(self.metrics[metric_name], list):
                    self.metrics[metric_name].append(value)
                elif isinstance(self.metrics[metric_name], dict):
                    if isinstance(value, tuple):
                        key, val = value
                        self.metrics[metric_name][key] = val
                else:
                    self.metrics[metric_name] += value
    
    def get_metrics(self) -> Dict:
        with self.lock:
            return self.metrics.copy()
    
    def export_metrics(self, filepath: str = "metrics.json"):
        with open(filepath, 'w') as f:
            json.dump(self.get_metrics(), f, indent=2)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('sebastian_multi_agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# MAIN ENTRY POINT

SORTING_CONFIG_FILE = "sorting_rules.json" 
WATCH_FOLDER = r"E:\Sebas" 
CONFIG_FILE = "sebastian_config.json"
MEMORY_BANK_FILE = "memory_bank.json"
SESSION_FILE = "sessions.json"

def load_config():
    defaults = {
        "stability_threshold": 10.0, 
        "analysis_cooldown": 5.0,
        "window_position": [1300, 50],
        "window_size": [500, 600],
        "screen_check_interval": 0.15,
        "screenshot_quality": (100, 100),
        "change_threshold": 0.005,
        "auto_edit_enabled": True,
        "save_folder": "./sebastian_fixes/",
        "confirm_apply": True,
        "detect_editor": "vscode",
        "max_memory_entries": 100,
        "context_window_size": 50000,
        "enable_mcp": True,
        "enable_parallel_agents": True,
        "agent_timeout": 30,
        "voice_enabled": True
    }
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return {**defaults, **json.load(f)}
    except Exception as e:
        logging.warning(f"Config load failed: {e}")
    return defaults

config = load_config()
os.makedirs(config["save_folder"], exist_ok=True)

metrics = MetricsCollector()

# API CONFIGURATION

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY:
    logging.error("API Key not found")
    print("\n❌ SETUP ERROR: Create a .env file with GOOGLE_API_KEY='your_key'")

try:
    if API_KEY:
        genai.configure(api_key=API_KEY)
        MODEL_NAME = 'models/gemini-2.0-flash'
        model = genai.GenerativeModel(MODEL_NAME)
        logging.info(f"API configured: {MODEL_NAME}")
except Exception as e:
    logging.error(f"API config failed: {e}")

# WATCHDOG & FILE SORTING SYSTEM

import shutil 

observer = None
watchdog_thread = None
watchdog_running = False 

class FileOrganizerHandler(FileSystemEventHandler):
    def __init__(self):
        self.rules = self.load_rules()
        self.processing_lock = threading.Lock()

    def load_rules(self):
        if os.path.exists(SORTING_CONFIG_FILE):
            try:
                with open(SORTING_CONFIG_FILE, 'r') as f:
                    rules = json.load(f)
                print(f"✅ Rules Loaded: {len(rules)} categories.")
                return rules
            except Exception as e:
                print(f"❌ Error reading JSON: {e}")
                return {}
        else:
            print(f"⚠️ {SORTING_CONFIG_FILE} NOT FOUND. Sorting will not work.")
            return {}

    # Triggers for any event
    def on_created(self, event):
        if not event.is_directory: self.organize_file(event.src_path, "Created")
    
    def on_moved(self, event):
        if not event.is_directory: self.organize_file(event.dest_path, "Renamed")
        
    def on_modified(self, event):
        if not event.is_directory: self.organize_file(event.src_path, "Modified")

    def organize_file(self, filepath, event_type):
        filename = os.path.basename(filepath)

        if filename.startswith("~") or filename.endswith(".tmp") or filename == "sorting_rules.json":
            return

        print(f"\n🔔 EVENT: {event_type} detected on: {filename}")

        if not self.rules:
            print("❌ STOPPING: No sorting rules loaded.")
            return

        try:
            time.sleep(1.0)
            
            if not os.path.exists(filepath):
                print("❌ STOPPING: File disappeared before processing.")
                return

            
            if "new text document" in filename.lower() or "new folder" in filename.lower():
                print("⏳ PAUSED: Ignoring 'New Text Document' (waiting for rename)")
                return

            
            base_dir = os.path.dirname(filepath)
            norm_base = os.path.normpath(base_dir).lower()
            norm_watch = os.path.normpath(WATCH_FOLDER).lower()

            if norm_base != norm_watch:
                print(f"⛔ IGNORED: File is in a subfolder, not the root.")
                print(f"   File location: {norm_base}")
                print(f"   Watch folder:  {norm_watch}")
                return

            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            filename_lower = filename.lower()
            
            target_folder = "Others"
            found_match = False

            for folder, criteria_list in self.rules.items():
                for criteria in criteria_list:
                    criteria = criteria.lower()
                    if criteria.startswith('.'):
                        if ext == criteria:
                            target_folder = folder
                            found_match = True
                            print(f"   MATCH FOUND: Extension {ext} -> {folder}")
                            break
                    else:
                        if criteria in filename_lower:
                            target_folder = folder
                            found_match = True
                            print(f"   MATCH FOUND: Keyword '{criteria}' -> {folder}")
                            break
                if found_match: break
            
            if not found_match:
                print(f"   NO MATCH: Defaulting to 'Others'")

            target_path = os.path.join(base_dir, target_folder)
            os.makedirs(target_path, exist_ok=True)
            dest_path = os.path.join(target_path, filename)

            base_name, extension = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(target_path, f"{base_name}_{counter}{extension}")
                counter += 1

            shutil.move(filepath, dest_path)
            print(f"✅ SUCCESS: Moved to {target_folder}")

        except Exception as e:
            print(f"❌ ERROR: {e}")

def watchdog_worker():
    """Background thread worker that keeps observer alive"""
    global observer, watchdog_running
    
    try:
        event_handler = FileOrganizerHandler()
        
        if not event_handler.rules:
            logging.error("❌ No sorting rules loaded - watchdog cannot start")
            print("❌ File sorting disabled: sorting_rules.json not found or empty")
            watchdog_running = False
            return
        
        observer = Observer()
        observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
        observer.start()
        
        logging.info(f"✅ Watchdog active: {WATCH_FOLDER}")
        print(f"👀 Watching: {WATCH_FOLDER}")
        print(f"📋 Loaded {len(event_handler.rules)} sorting categories")
        
        while watchdog_running:
            time.sleep(1)
        
        observer.stop()
        observer.join()
        logging.info("🛑 Watchdog stopped gracefully")
        print("🛑 File sorting stopped")
        
    except Exception as e:
        logging.error(f"❌ Watchdog thread error: {e}", exc_info=True)
        print(f"❌ Watchdog failed: {e}")
        watchdog_running = False

def start_watchdog():
    """Starts the file system observer in a background thread"""
    global watchdog_thread, watchdog_running

    if not os.path.exists(WATCH_FOLDER):
        try:
            os.makedirs(WATCH_FOLDER)
            print(f"📁 Created folder: {WATCH_FOLDER}")
            logging.info(f"Created watch folder: {WATCH_FOLDER}")
        except Exception as e:
            print(f"❌ Could not create folder {WATCH_FOLDER}: {e}")
            logging.error(f"Failed to create watch folder: {e}")
            return False

    if watchdog_running:
        logging.warning("⚠️ Watchdog already running")
        return True

    watchdog_running = True
    watchdog_thread = threading.Thread(target=watchdog_worker, daemon=True, name="WatchdogThread")
    watchdog_thread.start()

    time.sleep(0.5)
    
    return watchdog_running

def stop_watchdog():
    """Stops the watchdog observer gracefully"""
    global watchdog_running, observer
    
    if not watchdog_running:
        return
    
    logging.info("Stopping watchdog...")
    watchdog_running = False
    
    if watchdog_thread and watchdog_thread.is_alive():
        watchdog_thread.join(timeout=3)
    
    logging.info("Watchdog stopped")

# VOICE ASSISTANT SYSTEM

class VoiceHandler:
    """Handles Text-to-Speech and Speech-to-Text operations"""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.is_listening = False
        self.setup_voice()

    def setup_voice(self):
        """Configure TTS engine for Sebastian's persona"""
        try:
            voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', voices[0].id) 
            self.engine.setProperty('rate', 160)
            self.engine.setProperty('volume', 0.9)
        except Exception as e:
            logging.error(f"Voice setup error: {e}")

    def speak(self, text):
        """Non-blocking speak function"""
        def _speak_thread():
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[0].id)
                engine.setProperty('rate', 160)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logging.error(f"Speech error: {e}")
        
        threading.Thread(target=_speak_thread, daemon=True).start()

    def listen_for_command(self):
        """One-shot listening for commands"""
        with sr.Microphone() as source:
            logging.info("Listening for voice command...")
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.recognizer.pause_threshold = 1.0
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=15)
                self.recognizer.energy_threshold = 4000
                self.recognizer.dynamic_energy_threshold = True
                
                command = self.recognizer.recognize_google(audio)
                return command.lower()
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None
            except Exception as e:
                logging.error(f"Listen error: {e}")
                return None

voice_handler = VoiceHandler()

# AGENT TYPES & ENUMS

class AgentType(Enum):
    CODE_ANALYZER = "code_analyzer"
    BUG_DETECTOR = "bug_detector"
    FIX_GENERATOR = "fix_generator"
    SECURITY_AUDITOR = "security_auditor"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    DOCUMENTATION_WRITER = "documentation_writer"
    ORCHESTRATOR = "orchestrator"

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

# MEMORY BANK - Long Term Memory

class MemoryBank:
    """Persistent long-term memory storage for agents"""
    def __init__(self, filepath: str = MEMORY_BANK_FILE):
        self.filepath = filepath
        self.memories: List[Dict] = []
        self.load()
    
    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.memories = json.load(f)
                logging.info(f"Loaded {len(self.memories)} memories")
            except Exception as e:
                logging.error(f"Failed to load memory bank: {e}")
    
    def save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.memories, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save memory bank: {e}")
    
    def add_memory(self, content: str, tags: List[str], metadata: Dict = None):
        memory = {
            "id": str(uuid.uuid4()),
            "content": content,
            "tags": tags,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.memories.append(memory)
        if len(self.memories) > config["max_memory_entries"]:
            self.memories = self.memories[-config["max_memory_entries"]:]
        self.save()
        logging.info(f"Memory added: {memory['id']}")
    
    def search(self, query: str = None, tags: List[str] = None, limit: int = 10) -> List[Dict]:
        results = self.memories
        
        if tags:
            results = [m for m in results if any(tag in m.get('tags', []) for tag in tags)]
        
        if query:
            results = [m for m in results if query.lower() in m['content'].lower()]
        
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_context_summary(self, limit: int = 5) -> str:
        recent = self.memories[-limit:]
        summary = "Recent Context:\n"
        for mem in recent:
            summary += f"- [{', '.join(mem['tags'])}] {mem['content'][:100]}...\n"
        return summary

# SESSION MANAGEMENT

@dataclass
class Session:
    session_id: str
    created_at: datetime
    last_active: datetime
    state: Dict = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    
class InMemorySessionService:
    """Manages agent sessions and state"""
    def __init__(self, filepath: str = SESSION_FILE):
        self.filepath = filepath
        self.sessions: Dict[str, Session] = {}
        self.current_session_id: Optional[str] = None
        self.load()
    
    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    for sid, sdata in data.items():
                        self.sessions[sid] = Session(
                            session_id=sid,
                            created_at=datetime.fromisoformat(sdata['created_at']),
                            last_active=datetime.fromisoformat(sdata['last_active']),
                            state=sdata.get('state', {}),
                            history=sdata.get('history', [])
                        )
                logging.info(f"Loaded {len(self.sessions)} sessions")
            except Exception as e:
                logging.error(f"Failed to load sessions: {e}")
    
    def save(self):
        try:
            data = {}
            for sid, session in self.sessions.items():
                data[sid] = {
                    'created_at': session.created_at.isoformat(),
                    'last_active': session.last_active.isoformat(),
                    'state': session.state,
                    'history': session.history
                }
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save sessions: {e}")
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(
            session_id=session_id,
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        self.current_session_id = session_id
        self.save()
        logging.info(f"Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, state: Dict = None, history_entry: Dict = None):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_active = datetime.now()
            if state:
                session.state.update(state)
            if history_entry:
                session.history.append(history_entry)
            self.save()
    
    def pause_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].state['paused'] = True
            self.save()
    
    def resume_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].state['paused'] = False
            self.save()

# CONTEXT ENGINEERING - Context Compaction

class ContextManager:
    """Manages context window and performs compaction"""
    def __init__(self, max_tokens: int = 50000):
        self.max_tokens = max_tokens
        self.context_buffer = deque(maxlen=100)
    
    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def add_context(self, content: str, priority: int = 1):
        self.context_buffer.append({
            'content': content,
            'priority': priority,
            'timestamp': time.time(),
            'tokens': self.estimate_tokens(content)
        })
    
    def compact_context(self) -> str:
        """Intelligently compact context to fit within token limit"""
        total_tokens = sum(item['tokens'] for item in self.context_buffer)
        
        if total_tokens <= self.max_tokens:
            return "\n".join(item['content'] for item in self.context_buffer)
        
        sorted_items = sorted(
            self.context_buffer,
            key=lambda x: (x['priority'], x['timestamp']),
            reverse=True
        )
        
        compacted = []
        current_tokens = 0
        
        for item in sorted_items:
            if current_tokens + item['tokens'] <= self.max_tokens * 0.9:
                compacted.append(item['content'])
                current_tokens += item['tokens']
            else:
                remaining = [i['content'] for i in sorted_items if i not in compacted]
                if remaining:
                    summary = f"[Compacted {len(remaining)} older entries]"
                    compacted.append(summary)
                break
        
        return "\n".join(compacted)

# TOOLS SYSTEM

class Tool:
    """Base class for agent tools"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"Tool.{name}")
    
    async def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError

class FileAnalyzerTool(Tool):
    def __init__(self):
        super().__init__("file_analyzer", "Analyzes code files for issues")
    
    async def execute(self, filepath: str) -> Dict:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            metrics.record_metric("tool_usage", (self.name, 1))
            
            return {
                "filepath": filepath,
                "lines": len(lines),
                "size": len(content),
                "content": content[:5000]
            }
        except Exception as e:
            self.logger.error(f"File analysis failed: {e}")
            return {"error": str(e)}

class CodeExecutionTool(Tool):
    def __init__(self):
        super().__init__("code_execution", "Executes safe code snippets")
    
    async def execute(self, code: str, language: str = "python") -> Dict:
        self.logger.info(f"Code execution requested: {language}")
        metrics.record_metric("tool_usage", (self.name, 1))
        return {"status": "simulated", "output": "Execution feature pending"}

class WebSearchTool(Tool):
    def __init__(self):
        super().__init__("web_search", "Searches web for solutions")
    
    async def execute(self, query: str) -> Dict:
        self.logger.info(f"Web search: {query}")
        metrics.record_metric("tool_usage", (self.name, 1))
        return {"query": query, "results": ["Placeholder result"]}

class ToolRegistry:
    """Central registry for all available tools"""
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.register_default_tools()
    
    def register_default_tools(self):
        self.register(FileAnalyzerTool())
        self.register(CodeExecutionTool())
        self.register(WebSearchTool())
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        logging.info(f"Tool registered: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)
    
    async def execute_tool(self, name: str, *args, **kwargs) -> Any:
        tool = self.get_tool(name)
        if tool:
            return await tool.execute(*args, **kwargs)
        raise ValueError(f"Tool not found: {name}")

# AGENT BASE CLASS

@dataclass
class AgentTask:
    task_id: str
    agent_type: AgentType
    input_data: Dict
    created_at: float
    status: AgentStatus = AgentStatus.IDLE
    result: Optional[Dict] = None
    error: Optional[str] = None

class BaseAgent:
    """Base class for all specialized agents"""
    def __init__(self, agent_type: AgentType, tools: ToolRegistry):
        self.agent_type = agent_type
        self.agent_id = str(uuid.uuid4())
        self.status = AgentStatus.IDLE
        self.tools = tools
        self.logger = logging.getLogger(f"Agent.{agent_type.value}")
        self.context_manager = ContextManager()
    
    async def execute(self, task: AgentTask, memory: MemoryBank) -> Dict:
        """Execute agent task with observability"""
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        task.status = AgentStatus.RUNNING
        
        self.logger.info(f"Starting task: {task.task_id}")
        metrics.record_metric("agent_calls", 1)
        
        try:
            context = memory.get_context_summary()
            self.context_manager.add_context(context, priority=2)
            
            result = await self._process(task)
            
            task.status = AgentStatus.COMPLETED
            task.result = result
            self.status = AgentStatus.COMPLETED
            
            elapsed = time.time() - start_time
            metrics.record_metric("avg_response_time", elapsed)
            metrics.record_metric("agent_performance", (self.agent_type.value, elapsed))
            metrics.record_metric("successful_analyses", 1)
            
            self.logger.info(f"Task completed: {task.task_id} in {elapsed:.2f}s")
            
            memory.add_memory(
                content=f"Agent {self.agent_type.value} completed task",
                tags=[self.agent_type.value, "success"],
                metadata={"task_id": task.task_id, "duration": elapsed}
            )
            
            return result
            
        except Exception as e:
            task.status = AgentStatus.FAILED
            task.error = str(e)
            self.status = AgentStatus.FAILED
            metrics.record_metric("failed_analyses", 1)
            
            self.logger.error(f"Task failed: {task.task_id} - {e}")
            
            memory.add_memory(
                content=f"Agent {self.agent_type.value} failed: {str(e)}",
                tags=[self.agent_type.value, "failure"],
                metadata={"task_id": task.task_id, "error": str(e)}
            )
            
            raise
    
    async def _process(self, task: AgentTask) -> Dict:
        """Override in subclasses"""
        raise NotImplementedError

# SPECIALIZED AGENTS

class CodeAnalyzerAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__(AgentType.CODE_ANALYZER, tools)
    
    async def _process(self, task: AgentTask) -> Dict:
        filepath = task.input_data.get('filepath')
        screenshot = task.input_data.get('screenshot')
        
        if filepath:
            file_data = await self.tools.execute_tool('file_analyzer', filepath)
            analysis = f"Analyzed file: {filepath}\nLines: {file_data.get('lines', 0)}"
        else:
            analysis = "Screen-based analysis"
        
        return {
            "agent": self.agent_type.value,
            "analysis": analysis,
            "recommendations": ["Code structure looks good", "Consider adding error handling"]
        }

class BugDetectorAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__(AgentType.BUG_DETECTOR, tools)
    
    async def _process(self, task: AgentTask) -> Dict:
        code_content = task.input_data.get('code', '')
        filepath = task.input_data.get('filepath', 'Unknown')
        screenshot_path = task.input_data.get('screenshot_path')
        
        bugs = []
        
        try:
            prompt = f"""Analyze this code and find ALL bugs. For EACH bug, provide:
1. Exact line number
2. The CURRENT buggy line
3. The CORRECTED line (exact replacement)

File: {filepath}
Code:
{code_content}

Respond in this EXACT JSON format:
{{
    "bugs": [
        {{
            "line_number": <number>,
            "buggy_line": "<exact current line>",
            "fixed_line": "<exact corrected line>",
            "issue": "<brief 1-sentence issue>"
        }}
    ]
}}

Focus on: syntax errors, wrong keywords, wrong method names, logic errors.
BE PRECISE with line numbers and exact code."""

            if screenshot_path and os.path.exists(screenshot_path):
                img = Image.open(screenshot_path)
                response = model.generate_content([prompt, img])
            else:
                response = model.generate_content(prompt)
            
            result_text = response.text.strip()
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
                bugs = result_json.get('bugs', [])
            
            if not bugs:
                bugs = self._detect_bugs_from_code(code_content, filepath)
            
            self.logger.info(f"Bug detector found {len(bugs)} issues")
            
        except Exception as e:
            self.logger.error(f"Bug detection failed: {e}")
            bugs = self._detect_bugs_from_code(code_content, filepath)
        
        return {
            "agent": self.agent_type.value,
            "bugs_found": len(bugs),
            "bugs": bugs
        }
    
    def _detect_bugs_from_code(self, code: str, filepath: str) -> List[Dict]:
        """Pattern-based bug detection with line-by-line fixes"""
        bugs = []
        lines = code.split('\n')
    
        # Handle None filepath
        if not filepath:
            filepath = "unknown"
    
        is_java = filepath.endswith('.java')
        is_python = filepath.endswith('.py')
        
        for i, line in enumerate(lines, 1):
            original_line = line
            fixed_line = None
            issue = None
            
            if is_java:
                if 'public void main' in line and 'static' not in line:
                    fixed_line = line.replace('public void main', 'public static void main')
                    issue = "main method must be static"
                
                elif '.length()' in line:
                    fixed_line = line.replace('.length()', '.length')
                    issue = "Arrays use .length property"
                
                elif 'printLine' in line:
                    fixed_line = line.replace('printLine', 'println')
                    issue = "Wrong method name"
                
                elif '<=' in line and '.length' in line and 'for' in line:
                    fixed_line = line.replace('<=', '<')
                    issue = "Loop bound should use < not <="
            
            if is_python:
                if 'def init' in line:
                    fixed_line = line.replace('def init', 'def __init__')
                    issue = "Incorrect constructor name"
            
            if fixed_line and fixed_line != original_line:
                bugs.append({
                    "line_number": i,
                    "buggy_line": original_line.strip(),
                    "fixed_line": fixed_line.strip(),
                    "issue": issue
                })
        
        return bugs

class FixGeneratorAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__(AgentType.FIX_GENERATOR, tools)
    
    async def _process(self, task: AgentTask) -> Dict:
        bugs = task.input_data.get('bugs', [])
        
        previous_result = task.input_data.get('previous_result', {})
        if not bugs and previous_result:
            bugs = previous_result.get('bugs', [])
        
        if not bugs:
            self.logger.warning("No bugs provided to fix generator")
            return {
                "agent": self.agent_type.value,
                "fixes": [],
                "message": "No bugs to fix"
            }
        
        fixes = []
        for bug in bugs:
            fixes.append({
                "line_number": bug.get('line_number', bug.get('line', 'unknown')),
                "buggy_line": bug.get('buggy_line', ''),
                "fixed_line": bug.get('fixed_line', ''),
                "issue": bug.get('issue', bug.get('message', 'Fix applied'))
            })
        
        self.logger.info(f"Prepared {len(fixes)} fixes for application")
        
        return {
            "agent": self.agent_type.value,
            "fixes": fixes,
            "fixes_count": len(fixes)
        }

class SecurityAuditorAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        super().__init__(AgentType.SECURITY_AUDITOR, tools)
    
    async def _process(self, task: AgentTask) -> Dict:
        code = task.input_data.get('code', '')
        
        vulnerabilities = []
        if 'eval(' in code:
            vulnerabilities.append("Dangerous eval() usage detected")
        if 'exec(' in code:
            vulnerabilities.append("Dangerous exec() usage detected")
        
        return {
            "agent": self.agent_type.value,
            "vulnerabilities": vulnerabilities,
            "risk_level": "medium" if vulnerabilities else "low"
        }

# ORCHESTRATOR - Multi-Agent Coordination

class AgentOrchestrator:
    """Coordinates multiple agents in parallel and sequential workflows"""
    def __init__(self, tools: ToolRegistry, memory: MemoryBank, session_service: InMemorySessionService):
        self.tools = tools
        self.memory = memory
        self.session_service = session_service
        self.agents: Dict[AgentType, BaseAgent] = {}
        self.logger = logging.getLogger("Orchestrator")
        self.task_queue = asyncio.Queue()
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize all specialized agents"""
        self.agents[AgentType.CODE_ANALYZER] = CodeAnalyzerAgent(self.tools)
        self.agents[AgentType.BUG_DETECTOR] = BugDetectorAgent(self.tools)
        self.agents[AgentType.FIX_GENERATOR] = FixGeneratorAgent(self.tools)
        self.agents[AgentType.SECURITY_AUDITOR] = SecurityAuditorAgent(self.tools)
        
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    async def execute_parallel(self, tasks: List[AgentTask]) -> List[Dict]:
        """Execute multiple agents in parallel"""
        self.logger.info(f"Executing {len(tasks)} tasks in parallel")
        
        async def run_task(task):
            agent = self.agents.get(task.agent_type)
            if agent:
                return await agent.execute(task, self.memory)
            return {"error": "Agent not found"}
        
        results = await asyncio.gather(*[run_task(task) for task in tasks], return_exceptions=True)
        return [r if not isinstance(r, Exception) else {"error": str(r)} for r in results]
    
    async def execute_sequential(self, tasks: List[AgentTask]) -> List[Dict]:
        """Execute agents in sequence, passing output to next"""
        self.logger.info(f"Executing {len(tasks)} tasks sequentially")
        results = []
        previous_result = {}
        
        for task in tasks:
            task.input_data['previous_result'] = previous_result
            
            agent = self.agents.get(task.agent_type)
            if agent:
                result = await agent.execute(task, self.memory)
                results.append(result)
                previous_result = result
            else:
                results.append({"error": "Agent not found"})
        
        return results
    
    async def execute_workflow(self, workflow_type: str, input_data: Dict) -> Dict:
        """Execute a predefined workflow"""
        if workflow_type == "full_analysis":
            return await self._full_analysis_workflow(input_data)
        elif workflow_type == "security_audit":
            return await self._security_audit_workflow(input_data)
        else:
            raise ValueError(f"Unknown workflow: {workflow_type}")
    
    async def _full_analysis_workflow(self, input_data: Dict) -> Dict:
        """Complete code analysis workflow"""
        parallel_tasks = [
            AgentTask(
                task_id=str(uuid.uuid4()),
                agent_type=AgentType.CODE_ANALYZER,
                input_data=input_data,
                created_at=time.time()
            )
        ]
        
        parallel_results = await self.execute_parallel(parallel_tasks)
        
        sequential_tasks = [
            AgentTask(
                task_id=str(uuid.uuid4()),
                agent_type=AgentType.BUG_DETECTOR,
                input_data=input_data,
                created_at=time.time()
            ),
            AgentTask(
                task_id=str(uuid.uuid4()),
                agent_type=AgentType.FIX_GENERATOR,
                input_data={},
                created_at=time.time()
            )
        ]
        
        sequential_results = await self.execute_sequential(sequential_tasks)
        
        return {
            "workflow": "full_analysis",
            "parallel_results": parallel_results,
            "sequential_results": sequential_results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _security_audit_workflow(self, input_data: Dict) -> Dict:
        """Security-focused audit workflow"""
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.SECURITY_AUDITOR,
            input_data=input_data,
            created_at=time.time()
        )
        
        agent = self.agents[AgentType.SECURITY_AUDITOR]
        result = await agent.execute(task, self.memory)
        
        return {
            "workflow": "security_audit",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

# MAIN BOT STATE WITH MULTI-AGENT SUPPORT

class BotState:
    def __init__(self):
        self.status = "Idle"
        self.advice = "At your service, my lord."
        self.is_analyzing = False
        self.last_screen = None
        self.last_change_time = time.time()
        self.waiting_for_analysis = True
        self.is_paused = False
        self.countdown_text = ""
        self.last_analysis_time = 0
        self.analysis_cooldown = config["analysis_cooldown"]
        self.screen_hash_cache = {}
        self.lock = threading.Lock()
        self.detected_file_path = None
        self.parsed_fixes = []
        self.last_fix_file = None
        self.tools = ToolRegistry()
        self.memory = MemoryBank()
        self.session_service = InMemorySessionService()
        self.orchestrator = AgentOrchestrator(self.tools, self.memory, self.session_service)
        self.current_session = self.session_service.create_session()

state = BotState()

# CORE FUNCTIONS

def get_screen_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()

def detect_vscode_file():
    try:
        active_window = pyautogui.getActiveWindow()
        if active_window and "Visual Studio Code" in active_window.title:
            title = active_window.title
            filename_match = re.search(r'([^\[\\]+)\.(\w+)(?=\s*[\[])', title)
            if filename_match:
                filename = filename_match.group(1) + "." + filename_match.group(2)
                state.detected_file_path = os.path.join(os.getcwd(), filename)
                return state.detected_file_path
    except:
        pass
    return None

def parse_code_fixes(advice_text):
    """Extract fixes from analysis report"""
    fixes = []
    pattern = r'Line (\d+):\s*❌\s*(.+?)\s*✅\s*(.+?)(?=Line \d+:|$)'
    matches = re.findall(pattern, advice_text, re.DOTALL)
    
    for line_num, buggy, fixed in matches:
        fixes.append({
            "line_number": int(line_num),
            "buggy_line": buggy.strip(),
            "fixed_line": fixed.strip()
        })
    
    return fixes

def apply_fixes_to_file(fixes, target_file):
    """Apply fixes by REPLACING buggy lines in the original file"""
    if not fixes or not target_file or not os.path.exists(target_file):
        logging.warning(f"Cannot apply fixes: fixes={len(fixes) if fixes else 0}, file_exists={os.path.exists(target_file) if target_file else False}")
        return False
    
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        backup_file = target_file + '.backup'
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        fixed_count = 0
        
        for fix in fixes:
            if isinstance(fix, dict):
                line_num = fix.get('line_number')
                fixed_line = fix.get('fixed_line', '')
                
                if line_num and isinstance(line_num, int) and 1 <= line_num <= len(lines):
                    original = lines[line_num - 1]
                    indent = len(original) - len(original.lstrip())
                    
                    fixed_with_indent = ' ' * indent + fixed_line.strip() + '\n'
                    lines[line_num - 1] = fixed_with_indent
                    fixed_count += 1
                    
                    logging.info(f"Fixed line {line_num}: {fixed_line.strip()}")
        
        file_root, file_ext = os.path.splitext(target_file)
        fixed_file = f"{file_root}_fixed{file_ext}"
        
        with open(fixed_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state.last_fix_file = fixed_file
        
        logging.info(f"Applied {fixed_count} fixes to {fixed_file}")
        
        state.memory.add_memory(
            content=f"Applied {fixed_count} fixes to {os.path.basename(fixed_file)}",
            tags=["fix", "file_operation", "success"],
            metadata={
                "file": fixed_file, 
                "fix_count": fixed_count, 
                "timestamp": timestamp,
                "backup": backup_file
            }
        )
        
        return True, fixed_file
        
    except Exception as e:
        logging.error(f"Failed to apply fixes: {e}")
        state.memory.add_memory(
            content=f"Failed to apply fixes: {str(e)}",
            tags=["fix", "file_operation", "failure"],
            metadata={"file": target_file, "error": str(e)}
        )
        return False, None

async def analyze_with_agents(manual_trigger=False, user_query=None):
    """Multi-agent analysis function with Persona"""
    with state.lock:
        if state.is_analyzing:
            return
        
        if not manual_trigger and not user_query:
            time_since_last = time.time() - state.last_analysis_time
            if time_since_last < state.analysis_cooldown:
                state.countdown_text = f"Cooldown {round(state.analysis_cooldown - time_since_last, 1)}s"
                return
        
        state.is_analyzing = True
        state.status = "Multi-Agent Analysis..."
        state.countdown_text = "Agents working..."
    
    snap_path = "temp_snap.png"
    
    try:
        screenshot = pyautogui.screenshot()
        screen_hash = get_screen_hash(screenshot)
        
        if config["detect_editor"] == "vscode":
            state.detected_file_path = detect_vscode_file()
        
        if not user_query and screen_hash in state.screen_hash_cache and not manual_trigger:
            state.status = "Insight Ready (Cached)"
            state.advice = state.screen_hash_cache[screen_hash]
            state.is_paused = True
            metrics.record_metric("cache_hits", 1)
            logging.info("Returned cached result")
            return
        
        metrics.record_metric("cache_misses", 1)
        screenshot.save(snap_path)
        
        input_data = {
            "screenshot_path": snap_path,
            "filepath": state.detected_file_path,
            "user_query": user_query,
            "timestamp": datetime.now().isoformat()
        }
        
        if state.detected_file_path and os.path.exists(state.detected_file_path):
            with open(state.detected_file_path, 'r') as f:
                input_data["code"] = f.read()
        
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        
        if user_query:
            generate_keywords = ["write", "create", "generate", "make", "build", "code", "solution", "program"]
            is_code_generation = any(keyword in user_query.lower() for keyword in generate_keywords)
            
            if is_code_generation:
                prompt = f"""
Identity: You are Sebastian Michaelis, the butler. You are flawless, elegant, and highly skilled in programming.
Address the user as "Master" or "Young Master".

User Request: "{user_query}"
Time: {current_time}

Task: Write the COMPLETE, WORKING code solution for the user's request.
- Include ALL necessary imports
- Write PRODUCTION-READY code with proper error handling
- Add brief comments for complex logic
- Format code beautifully

Respond with:
1. A polite butler greeting (1-2 lines)
2. The complete code in a markdown code block
3. Brief usage instructions if needed
4. A closing statement

Example format:
"Certainly, Master. I shall craft this solution with utmost precision.

```python
# Your complete code here
```

This implementation should serve you well, young master."
"""
                
                response = model.generate_content(prompt)
                gemini_insight = response.text.strip()
                
                final_advice = f"=== Code Generation ===\nRequest: {user_query}\n\n{gemini_insight}"
                
                state.status = "Code Generated"
                state.advice = final_advice
                state.is_paused = True
                
                if config.get("voice_enabled", True):
                    speech_text = "Code generated successfully, my lord."
                    voice_handler.speak(speech_text)
                
                if "```" in gemini_insight:
                    code_match = re.search(r'```(\w+)?\n(.*?)\n```', gemini_insight, re.DOTALL)
                    if code_match:
                        generated_code = code_match.group(2)
                        
                        save_response = messagebox.askyesno(
                            "Save Generated Code?",
                            "Would you like to save this code to a file?"
                        )
                        
                        if save_response:
                            file_path = filedialog.asksaveasfilename(
                                defaultextension=".py",
                                filetypes=[("Python files", "*.py"), ("Java files", "*.java"), ("All files", "*.*")]
                            )
                            if file_path:
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(generated_code)
                                state.advice += f"\n\n✅ Code saved to {os.path.basename(file_path)}"
                                state.detected_file_path = file_path
                
                state.memory.add_memory(
                    content=f"Generated code for: {user_query}",
                    tags=["code_generation", "voice_command"],
                    metadata={"query": user_query}
                )
                
                return
        
        workflow_result = await state.orchestrator.execute_workflow("full_analysis", input_data)
        
        advice_parts = ["=== Multi-Agent Analysis Report ===\n"]
        all_bugs = []
        all_fixes = []
        
        for result in workflow_result.get("parallel_results", []):
            agent_name = result.get("agent", "Unknown")
            advice_parts.append(f"\n[{agent_name.upper()}]")
            if "analysis" in result:
                advice_parts.append(result["analysis"])
            if "vulnerabilities" in result:
                vuln_list = result['vulnerabilities']
                if vuln_list:
                    advice_parts.append(f"Vulnerabilities: {', '.join(vuln_list)}")
                else:
                    advice_parts.append("No security vulnerabilities found")
        
        state.parsed_fixes = []
        
        for result in workflow_result.get("sequential_results", []):
            agent_name = result.get("agent", "Unknown")
            
            if "bugs" in result:
                bugs = result.get("bugs", [])
                all_bugs.extend(bugs)
                bugs_count = result.get('bugs_found', len(bugs))
                
                if bugs:
                    advice_parts.append(f"\n[BUGS FOUND: {bugs_count}]")
                    for bug in bugs:
                        line_num = bug.get('line_number', 'unknown')
                        buggy = bug.get('buggy_line', '')
                        fixed = bug.get('fixed_line', '')
                        advice_parts.append(f"Line {line_num}:")
                        advice_parts.append(f"  Buggy: {buggy}")
                        advice_parts.append(f"  Fixed: {fixed}")
                        
                        state.parsed_fixes.append({
                            "line_number": line_num,
                            "buggy_line": buggy,
                            "fixed_line": fixed
                        })
            
            if "fixes" in result:
                fixes = result.get("fixes", [])
                all_fixes.extend(fixes)
                
                if fixes:
                    advice_parts.append(f"\n[READY TO APPLY: {len(fixes)} fixes]")
                    for fix_item in fixes:
                        if isinstance(fix_item, dict):
                            existing = next((f for f in state.parsed_fixes 
                                             if f.get('line_number') == fix_item.get('line_number')), None)
                            if not existing:
                                state.parsed_fixes.append(fix_item)
        
        logging.info(f"Stored {len(state.parsed_fixes)} fixes in state.parsed_fixes")
        final_advice = "\n".join(advice_parts)
        
        if user_query:
            prompt = f"""
Identity: You are Sebastian, Multi AI Agent System.
Address the user as "Master".

User Query: "{user_query}"
Context: Time is {current_time}.
File: {state.detected_file_path or 'No file open'}

Task: Answer the user's query politely and helpfully based on the screenshot context.
"""
        else:
            prompt = f"""
Identity: You are Sebastian, Multi AI Agent System.

Time: {current_time}
File: {state.detected_file_path or 'Unknown'}

Analysis Report:
{final_advice}

Task: Review the multi-agent analysis above. Provide your elegant summary and recommendations.
Be concise but insightful.
"""
        
        img_file = Image.open(snap_path)
        response = model.generate_content([prompt, img_file])
        gemini_insight = response.text.strip()
        
        final_advice += f"\n\n=== Sebastian's Insight ===\n{gemini_insight}"
        
        state.status = "Analysis Complete"
        state.advice = final_advice
        state.is_paused = True
        
        if config.get("voice_enabled", True):
            speech_lines = gemini_insight.replace("*", "").replace("`", "").split("\n")
            speech_text = " ".join([line for line in speech_lines[:3] if line.strip()])
            voice_handler.speak(speech_text[:200])
        
        if not user_query:
            state.screen_hash_cache[screen_hash] = final_advice
        
        state.session_service.update_session(
            state.current_session,
            state={"last_analysis": datetime.now().isoformat()},
            history_entry={
                "type": "analysis",
                "workflow": "full_analysis",
                "timestamp": datetime.now().isoformat(),
                "result_summary": "Multi-agent analysis completed"
            }
        )
        
        img_file.close()
        logging.info("Multi-agent analysis complete")
        
    except Exception as e:
        state.status = "Error"
        state.advice = f"Apologies, my lord.\n\nError: {str(e)}"
        logging.error(f"Analysis error: {e}")
    finally:
        if os.path.exists(snap_path):
            try:
                os.remove(snap_path)
            except:
                pass
        with state.lock:
            state.is_analyzing = False
            state.last_analysis_time = time.time()

def analyze_code_wrapper(manual_trigger=False, user_query=None):
    """Wrapper to run async analysis in thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(analyze_with_agents(manual_trigger, user_query))
    loop.close()

def on_auto_analyze():
    threading.Thread(target=analyze_code_wrapper, args=(True,), daemon=True).start()

def on_apply_fixes():
    pass 

keyboard.add_hotkey('ctrl+shift+a', on_auto_analyze)
keyboard.add_hotkey('ctrl+shift+m', lambda: metrics.export_metrics())

gui_root = None

def restore_window_hotkey():
    """Restore minimized window with Ctrl+Shift+R (Thread-Safe)"""
    global gui_root
    try:

        if gui_root and gui_root.winfo_exists():

            gui_root.after(0, lambda: gui_root.deiconify())
            gui_root.after(0, lambda: gui_root.state('normal'))
            gui_root.after(0, lambda: gui_root.lift())
            gui_root.after(0, lambda: gui_root.focus_force())
            
            logging.info("GUI restored from hotkey")
            print("GUI restored via Ctrl+Shift+R")
    except Exception as e:
        logging.error(f"Failed to restore window: {e}")

# SCREEN MONITORING

def check_screen_stability():
    logging.info("Screen monitor started")
    
    # We use a small size for comparison to ignore tiny noise (antialiasing/compression artifacts)
    COMPARE_SIZE = (100, 100) 
    
    while True:
        time.sleep(config["screen_check_interval"])
        
        if state.is_analyzing or state.is_paused:
            continue
        
        try:
            # Capture and resize immediately for "fuzzy" comparison
            # This acts as a low-pass filter to ignore single-pixel noise
            screenshot_raw = pyautogui.screenshot()
            current_screen = screenshot_raw.resize(COMPARE_SIZE)
            
            if state.last_screen:
                # Calculate the absolute difference between images
                diff = ImageChops.difference(current_screen, state.last_screen)
                
                # Get the 'energy' of the difference (Sum of all pixel differences)
                stat = ImageStat.Stat(diff)
                diff_value = sum(stat.mean) 
                
                # THRESHOLD TUNING:
                # 0.0 = Identical images
                # 1.0 = Tiny noise (like a blinking cursor in a small generic window)
                # >5.0 = Actual typing or scrolling
                
                # If the visual difference is significant enough:
                if diff_value > 2.5: 
                    state.last_change_time = time.time()
                    state.status = "Watching..."
                    state.waiting_for_analysis = True
            
            # Update last_screen reference
            state.last_screen = current_screen
            
            # Calculate elapsed time
            elapsed = time.time() - state.last_change_time
            
            if elapsed >= config["stability_threshold"]:
                state.countdown_text = "Ready"
                if state.waiting_for_analysis:
                    state.waiting_for_analysis = False
                    # Trigger analysis
                    threading.Thread(target=analyze_code_wrapper, daemon=True).start()
            else:
                if state.waiting_for_analysis:
                    remaining = config['stability_threshold'] - elapsed
                    state.countdown_text = f"{remaining:.1f}s"
                    
        except Exception as e:
            logging.error(f"Screen monitor error: {e}")

# GUI

def run_hud():
    root = tk.Tk()
    gui_root = root
    root.title("Sebastian - Multi-Agent AI System")
    
    pos = config["window_position"]
    size = config["window_size"]
    root.geometry(f"{size[0]}x{size[1]}+{pos[0]}+{pos[1]}")
    
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.95)
    root.configure(bg="#1e1e1e")
    root.overrideredirect(True)
    
    def start_move(event):
        if isinstance(event.widget, (tk.Text, tk.Entry, scrolledtext.ScrolledText)):
            root.dragging = False
            return
        root.x = event.x
        root.y = event.y
        root.dragging = True
    
    def do_move(event):
        if not getattr(root, 'dragging', False):
            return
        x = root.winfo_x() + (event.x - root.x)
        y = root.winfo_y() + (event.y - root.y)
        root.geometry(f"+{x}+{y}")
    
    root.bind("<Button-1>", start_move)
    root.bind("<B1-Motion>", do_move)
    
    resize_grip = tk.Label(root, text="◢", bg="#1e1e1e", fg="#555555", cursor="sizing", font=("Arial", 12))
    resize_grip.place(relx=1.0, rely=1.0, anchor="se")
    
    def do_resize(event):
        x = root.winfo_pointerx() - root.winfo_rootx()
        y = root.winfo_pointery() - root.winfo_rooty()
        x = max(350, x)
        y = max(300, y)
        root.geometry(f"{x}x{y}")
        return "break"
    
    resize_grip.bind("<B1-Motion>", do_resize)
    
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    
    header_frame = tk.Frame(root, bg="#1e1e1e")
    header_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))
    
    top_row = tk.Frame(header_frame, bg="#1e1e1e")
    top_row.pack(side="top", fill="x")
    
    lbl_greeting = tk.Label(
        top_row, text="Sebastian - Multi-Agent System", bg="#1e1e1e", fg="white",
        font=("Segoe UI", 11, "bold"), anchor="w"
    )
    lbl_greeting.pack(side="left", fill="x", expand=True)

    def minimize_window():
        root.withdraw()
        logging.info("GUI minimized - Sebastian still running in background")
        print("GUI minimized - Watchdog still active")

    btn_minimize = tk.Button(
        top_row, text="─", command=minimize_window, bg="#1e1e1e", fg="#feca57",
        font=("Arial", 12, "bold"), bd=0, activebackground="#feca57", width=2
    )
    btn_minimize.pack(side="right", padx=(5, 0))

    def graceful_exit():
        if messagebox.askyesno("Confirm Exit", 
                            "Close Sebastian GUI?\n\nFile sorting will continue in background."):
            metrics.export_metrics()
            state.memory.save()
            state.session_service.save()
            logging.info("GUI closed - Watchdog still active")
            print("GUI closed - File sorting continues")
            root.quit()
            root.destroy()

    btn_close = tk.Button(
        top_row, text="✕", command=graceful_exit, bg="#1e1e1e", fg="#ff6b6b",
        font=("Arial", 12, "bold"), bd=0, activebackground="#ff6b6b", width=2
    )
    btn_close.pack(side="right")
    
    status_container = tk.Frame(header_frame, bg="#1e1e1e")
    status_container.pack(side="top", fill="x", pady=(2, 0))
    
    lbl_status = tk.Label(status_container, text="Initializing...", bg="#1e1e1e", fg="#00ff9d", font=("Segoe UI", 9), anchor="w")
    lbl_status.pack(side="left")
    
    lbl_timer = tk.Label(status_container, text="--", bg="#1e1e1e", fg="#888888", font=("Consolas", 9), anchor="e")
    lbl_timer.pack(side="right")
    
    lbl_file = tk.Label(status_container, text="No file", bg="#1e1e1e", fg="#888888", font=("Consolas", 8))
    lbl_file.pack(side="right", padx=(5,0))
    
    agent_status_frame = tk.Frame(header_frame, bg="#1e1e1e")
    agent_status_frame.pack(side="top", fill="x", pady=(5, 0))
    
    lbl_agents = tk.Label(agent_status_frame, text="Agents: 4 Active", bg="#1e1e1e", fg="#888888", font=("Consolas", 8))
    lbl_agents.pack(side="left")
    
    lbl_session = tk.Label(agent_status_frame, text=f"Session: {state.current_session[:8]}", bg="#1e1e1e", fg="#888888", font=("Consolas", 8))
    lbl_session.pack(side="right")
    
    text_font = font.Font(family="Consolas", size=10)
    st_advice = scrolledtext.ScrolledText(
        root, wrap=tk.WORD, font=text_font, bg="#252526", fg="#d4d4d4",
        bd=0, padx=5, pady=5, selectbackground="#264f78", selectforeground="white"
    )
    st_advice.grid(row=1, column=0, sticky="nsew", padx=15, pady=5)

    def on_key(e):
        if e.keysym in ['Up', 'Down', 'Left', 'Right', 'Prior', 'Next', 'Home', 'End']:
            return None
        if e.state & 0x4: 
            return None
        return "break"
    
    st_advice.bind("<Key>", on_key)
    
    st_advice.insert(tk.END, "Multi-Agent System Initialized\n\nAt your service, my lord.")
    
    input_frame = tk.Frame(root, bg="#1e1e1e")
    input_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(10, 0))
    
    txt_input = tk.Entry(input_frame, bg="#3c3c3c", fg="white", font=("Segoe UI", 10), relief="flat", insertbackground="white")
    txt_input.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 5))
    
    def send_chat(event=None):
        query = txt_input.get().strip()
        if not query:
            return
        txt_input.delete(0, tk.END)
        st_advice.delete("1.0", tk.END)
        st_advice.insert(tk.END, "Agents analyzing...")
        threading.Thread(target=analyze_code_wrapper, args=(True, query), daemon=True).start()
    
    txt_input.bind("<Return>", send_chat)
    
    btn_send = tk.Button(input_frame, text="➤", command=send_chat, bg="#0984e3", fg="white", relief="flat", font=("Segoe UI", 10, "bold"))
    btn_send.pack(side="right")
    
    def start_voice_listen():
        state.status = "Listening..."
        def _listen():
            cmd = voice_handler.listen_for_command()
            if cmd:
                root.after(0, lambda: txt_input.insert(0, cmd))
                root.after(0, send_chat)
            else:
                def set_unrecognized():
                    state.status = "Voice unrecognized"
                root.after(0, set_unrecognized)
                
        threading.Thread(target=_listen, daemon=True).start()
    
    btn_voice = tk.Button(input_frame, text="🎤", command=start_voice_listen, bg="#e056fd", fg="white", relief="flat", font=("Segoe UI", 10))
    btn_voice.pack(side="right", padx=5)
    
    btn_frame = tk.Frame(root, bg="#1e1e1e")
    btn_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(10, 0))
    
    def toggle_pause():
        state.is_paused = not state.is_paused
        if state.is_paused:
            state.session_service.pause_session(state.current_session)
        else:
            state.session_service.resume_session(state.current_session)
    
    def reset_state():
        state.advice = "Monitoring..."
        state.status = "Watching..."
        state.waiting_for_analysis = True
        state.is_paused = False
    
    def force_analyze():
        state.status = "Queued..."
        st_advice.delete("1.0", tk.END)
        st_advice.insert(tk.END, "Multi-agent analysis starting...")
        threading.Thread(target=analyze_code_wrapper, args=(True,), daemon=True).start()
    
    def apply_fixes_gui():
        logging.info(f"Apply button clicked. Fixes available: {len(state.parsed_fixes)}, File: {state.detected_file_path}")
        
        if not state.parsed_fixes:
            messagebox.showinfo("Info", "No fixes available. Please analyze code first.")
            return
        
        if not state.detected_file_path:
            if messagebox.askyesno("File Not Detected", "Could not detect active file automatically.\n\nWould you like to select the file manually?"):
                file_path = filedialog.askopenfilename(
                    title="Select Source File to Fix",
                    filetypes=[("Code Files", "*.py;*.java;*.js;*.html;*.css;*.cpp"), ("All Files", "*.*")]
                )
                if file_path:
                    state.detected_file_path = file_path
                    lbl_file.config(text=os.path.basename(file_path))
                else:
                    return
            else:
                return
        
        if not os.path.exists(state.detected_file_path):
            messagebox.showerror("Error", f"File not found: {state.detected_file_path}")
            return
        
        success, fixed_path = apply_fixes_to_file(state.parsed_fixes, state.detected_file_path)
        if success:
            st_advice.insert(tk.END, f"\n\n✅ Applied {len(state.parsed_fixes)} fixes!\nSaved to: {os.path.basename(fixed_path)}")
            messagebox.showinfo("Success", f"Applied {len(state.parsed_fixes)} fixes!\n\nNew file created:\n{fixed_path}")
            state.parsed_fixes = []
        else:
            messagebox.showerror("Error", "Failed to apply fixes - check logs")
    
    def show_metrics():
            m = metrics.get_metrics()
            avg_time = sum(m['avg_response_time'])/len(m['avg_response_time']) if m['avg_response_time'] else 0
                
            metrics_window = tk.Toplevel(root)
            metrics_window.title("System Metrics")
            metrics_window.geometry("400x350")
            metrics_window.configure(bg="#1e1e1e")
                
            tk.Label(metrics_window, text="📊 System Metrics", bg="#1e1e1e", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=10)
                
            metrics_text = f"""
🤖 Agent Calls: {m['agent_calls']}
✅ Successful Analyses: {m['successful_analyses']}
❌ Failed Analyses: {m['failed_analyses']}
⚡ Avg Response Time: {avg_time:.2f}s
💾 Cache Hits: {m['cache_hits']}
🔍 Cache Misses: {m['cache_misses']}

🧠 Memory Entries: {len(state.memory.memories)}
📋 Active Session: {state.current_session[:8]}...
                """
                
            tk.Label(metrics_window, text=metrics_text, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 10), justify="left").pack(pady=10, padx=20)
                
            tk.Button(metrics_window, text="Export Metrics", command=lambda: [metrics.export_metrics(), messagebox.showinfo("Exported", "Metrics saved to metrics.json")], bg="#6c5ce7", fg="white", font=("Segoe UI", 10, "bold")).pack(pady=10)
            
    def show_memory():
        memories = state.memory.search(limit=15)
        
        memory_window = tk.Toplevel(root)
        memory_window.title("Memory Bank")
        memory_window.geometry("600x500")
        memory_window.configure(bg="#1e1e1e")
        
        tk.Label(memory_window, text="Memory Bank", bg="#1e1e1e", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=10)
        
        memory_text = scrolledtext.ScrolledText(memory_window, wrap=tk.WORD, bg="#252526", fg="#d4d4d4", font=("Consolas", 9), height=20)
        memory_text.pack(pady=10, padx=20, fill="both", expand=True)
        
        for mem in memories:
            memory_text.insert(tk.END, f"[{', '.join(mem['tags'])}]\n")
            memory_text.insert(tk.END, f"{mem['content'][:150]}...\n")
            memory_text.insert(tk.END, f"Time: {mem['timestamp']}\n")
            memory_text.insert(tk.END, "-" * 60 + "\n\n")
        
        memory_text.config(state=tk.DISABLED)

    
    btn_style = {"relief": "flat", "font": ("Segoe UI", 9, "bold")}
    
    btn_pause = tk.Button(btn_frame, text="Stop", command=toggle_pause, bg="#d63031", fg="white", width=6, **btn_style)
    btn_pause.pack(side="left", padx=(0, 2), fill="x", expand=True)
    
    btn_next = tk.Button(btn_frame, text="Reset", command=reset_state, bg="#4a4a4a", fg="white", width=6, **btn_style)
    btn_next.pack(side="left", padx=(2, 2), fill="x", expand=True)
    
    btn_analyze = tk.Button(btn_frame, text="Analyze", command=force_analyze, bg="#0984e3", fg="white", width=8, **btn_style)
    btn_analyze.pack(side="left", padx=(2, 2), fill="x", expand=True)
    
    btn_apply = tk.Button(btn_frame, text="Apply Fixes", command=apply_fixes_gui, bg="#00b894", fg="white", width=10, **btn_style)
    btn_apply.pack(side="left", padx=(2, 0), fill="x", expand=True)
    

    btn_frame2 = tk.Frame(root, bg="#1e1e1e")
    btn_frame2.grid(row=4, column=0, sticky="ew", padx=15, pady=(5, 15))
    
    btn_metrics = tk.Button(btn_frame2, text="📊 Metrics", command=show_metrics, bg="#6c5ce7", fg="white", **btn_style)
    btn_metrics.pack(side="left", padx=(0, 2), fill="x", expand=True)
    
    btn_memory = tk.Button(btn_frame2, text="🧠 Memory", command=show_memory, bg="#a29bfe", fg="white", **btn_style)
    btn_memory.pack(side="left", padx=(2, 0), fill="x", expand=True)
    
    lbl_countdown = tk.Label(root, text="", bg="#1e1e1e", fg="#666666", font=("Consolas", 8))
    lbl_countdown.place(relx=0.5, rely=1.0, anchor="s", y=-5)
    
    def update_ui():
        """Update UI labels periodically"""
        try:
            lbl_status.config(text=state.status)
            

            if state.is_analyzing or "Analysis" in state.status or "Queued" in state.status:
                lbl_status.config(fg="#f1c40f")
            elif "Listening" in state.status:
                lbl_status.config(fg="#00d2d3")
            elif "Error" in state.status:
                lbl_status.config(fg="#ff6b6b")
            else:
                lbl_status.config(fg="#00ff9d")

            current_text = state.advice
            if st_advice.get("1.0", tk.END).strip() != current_text.strip():
                st_advice.delete("1.0", tk.END)
                st_advice.insert(tk.END, current_text)
                
            lbl_timer.config(text=state.countdown_text)
            lbl_countdown.config(text=state.countdown_text)
            
            if state.detected_file_path:
                lbl_file.config(text=os.path.basename(state.detected_file_path))
            else:
                lbl_file.config(text="No file")
            
        except Exception as e:
            logging.error(f"UI update error: {e}")
        
        root.after(200, update_ui)
    
    threading.Thread(target=check_screen_stability, daemon=True).start()
    
    update_ui()
    
    logging.info("Sebastian GUI started")
    root.mainloop()

# MAIN ENTRY POINT

if __name__ == "__main__":
    try:
        logging.info("=" * 60)
        logging.info("SEBASTIAN MULTI-AGENT AI SYSTEM STARTING")
        logging.info("=" * 60)
        
        start_watchdog()

        logging.info(f"Session: {state.current_session}")
        logging.info(f"Agents: {len(state.orchestrator.agents)}")
        logging.info(f"Tools: {len(state.tools.tools)}")
        logging.info(f"Memory Entries: {len(state.memory.memories)}")
        logging.info("=" * 60)
        
        print("\n" + "=" * 60)
        print("🎩 SEBASTIAN MULTI-AGENT AI SYSTEM")
        print("=" * 60)
        print("Status: ✅ Initialized")
        print(f"Agents: {len(state.orchestrator.agents)} Active")
        print(f"File Sorting: ✅ Active (Background)")
        print(f"Session: {state.current_session[:16]}...")
        print("\n🔧 Hotkeys:")
        print("  Ctrl+Shift+A  - Force Analysis")
        print("  Ctrl+Shift+S  - Apply Fixes")
        print("  Ctrl+Shift+M  - Export Metrics")
        print("  Ctrl+Shift+R  - Restore Window")
        print("\n🎤 Voice: Enabled" if config.get("voice_enabled") else "\n🎤 Voice: Disabled")
        print("=" * 60 + "\n")
        
        run_hud()
        
        print("\n👋 GUI closed. File sorting still running in background.")
        print("Press Ctrl+C to stop Sebastian completely...")
        
        try:
            while watchdog_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping Sebastian...")
            stop_watchdog()
        
    except KeyboardInterrupt:
        logging.info("Shutdown requested by user")
        stop_watchdog()
        metrics.export_metrics()
        state.memory.save()
        state.session_service.save()
        print("\n👋 Sebastian shutting down gracefully...")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        stop_watchdog()
    finally:
        logging.info("Sebastian terminated")
        print("Goodbye, Master.")