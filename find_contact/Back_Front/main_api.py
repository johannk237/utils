# main_api.py
import time
import random
import os
import csv
import re
import logging
import shutil # For saving uploaded file
import uuid # For generating unique task IDs
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import List, Dict, Set, Tuple, Optional, Any
from pydantic import BaseModel # For text input model

# --- FastAPI Imports ---
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Path
from fastapi.middleware.cors import CORSMiddleware
import uvicorn # Required to run the app

# --- Dependency Checks & Imports (Copied from find_contact_from_name_oop.py) ---
# (Keep existing imports: logging, googleapiclient, langchain, playwright, pandas)
load_dotenv()

LOG_FILE = "main_processor_api.log" # New log file name
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s', # Added line number
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from googleapiclient.discovery import build, Resource
    GOOGLE_API_AVAILABLE = True
except ImportError: GOOGLE_API_AVAILABLE = False; logger.warning("google-api-python-client not found.")
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.exceptions import OutputParserException
    from langchain_core.language_models.chat_models import BaseChatModel
    LANGCHAIN_AVAILABLE = True
except ImportError: LANGCHAIN_AVAILABLE = False; logger.warning("LangChain/Groq not found.")
try:
    from playwright.sync_api import (
        sync_playwright, Playwright, Browser, Page, BrowserContext,
        TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError: PLAYWRIGHT_AVAILABLE = False; logger.warning("Playwright not found.")
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError: PANDAS_AVAILABLE = False; logger.critical("Pandas not found. Exiting."); exit()
# --------------------------------------------------------------------------

# ===========================================
# Task Status Management (NEW)
# ===========================================
# Simple in-memory storage for task statuses.
# In a production scenario, consider using Redis, a database, or Celery results backend.
task_statuses: Dict[str, Dict[str, Any]] = {}

def update_task_status(task_id: str, status: str, message: str, output_file: Optional[str] = None, log_file: Optional[str] = None, error_detail: Optional[str] = None):
    """Updates the status of a task."""
    if task_id not in task_statuses:
        logger.warning(f"Attempted to update status for unknown task_id: {task_id}")
        return
    task_statuses[task_id]['status'] = status
    task_statuses[task_id]['progress_message'] = message
    if output_file: task_statuses[task_id]['output_file'] = output_file
    if log_file: task_statuses[task_id]['log_file'] = log_file
    if error_detail: task_statuses[task_id]['error_detail'] = error_detail
    logger.debug(f"Task {task_id} status updated: {status} - {message}")

# ===========================================
# Configuration Class (Slightly Modified)
# ===========================================
class Config:
    """Holds configuration settings, now takes input file path dynamically."""
    # --- File Paths ---
    # INPUT_NAMES_FILE is now dynamic
    FINAL_OUTPUT_FILE_PREFIX: str = "profiles_output" # Prefix for output files
    # State Tracking
    GOOGLE_SEARCHED_NAMES_FILE: str = "processed_google_searched_names.txt"
    GOOGLE_SEARCH_NO_RESULTS_FILE: str = "processed_google_search_no_results.txt"
    PROCESSED_URL_ANALYSIS_FILE: str = "processed_url_profile_analysis.txt"
    TEMP_UPLOAD_DIR: str = "temp_uploads" # Directory for uploaded files
    TEMP_TEXT_INPUT_DIR: str = "temp_text_inputs" # Directory for text inputs

    # --- API Keys & IDs ---
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
    GOOGLE_CX_ID: Optional[str] = os.getenv("GOOGLE_CX_ID", "YOUR_CSE_ID_HERE")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

    # --- Google Search ---
    NUM_GOOGLE_RESULTS: int = 10

    # --- LLM ---
    LLM_FILTER_MODEL_NAME: str = os.getenv("LLM_FILTER_MODEL", "llama3-8b-8192")
    LLM_ANALYZE_MODEL_NAME: str = os.getenv("LLM_ANALYZE_MODEL", "llama3-70b-8192")

    # --- Playwright ---
    USER_AGENTS: List[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]
    NAVIGATION_TIMEOUT: int = 35_000
    PAGE_LOAD_TIMEOUT: int = 45_000
    MAX_CONTENT_LENGTH: int = 25000

    # --- Output ---
    OUTPUT_CSV_SEPARATOR: str = ";"
    PROFILE_FIELDS_MAPPING: Dict[str, str] = {
        "job_title": "JobTitle", "company": "Company", "location": "Location", "summary": "Summary",
        "associated_emails": "AssociatedEmails", "associated_phones": "AssociatedPhones"
    }
    CSV_HEADERS: List[str] = ["Nom", "WebsiteURL"] + list(PROFILE_FIELDS_MAPPING.values()) + ["LLMSource"]

    # --- Regex ---
    EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    PHONE_REGEX = re.compile(r'(?:(?:\+|00)\s*\d{1,3}[-.\s]?)?(?:\(\s*\d{1,4}\s*\)|(?:\d+))?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b')

    def __init__(self):
        # Ensure temp dirs exist
        os.makedirs(self.TEMP_UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.TEMP_TEXT_INPUT_DIR, exist_ok=True)
        # Validate essential configurations
        if not self.GROQ_API_KEY: logger.warning("Groq API Key missing.")
        if (not self.GOOGLE_API_KEY or self.GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE") or \
           (not self.GOOGLE_CX_ID or self.GOOGLE_CX_ID == "YOUR_CSE_ID_HERE"):
            logger.warning("Google API Key/CX ID missing.")

    def get_output_filename(self, task_id: str) -> str:
        """Generates a unique output filename based on task ID."""
        return f"{self.FINAL_OUTPUT_FILE_PREFIX}_{task_id}.csv"

# ===========================================
# State Manager Class (Unchanged)
# ===========================================
class StateManager:
    # (No changes needed here)
    def __init__(self, config: Config):
        self.config = config
        self.google_searched_names: Set[str] = set()
        self.google_search_no_results: Set[str] = set()
        self.processed_url_analysis: Set[str] = set()
        self.names_to_skip_completely: Set[str] = set()
        self._load_all_states()
    def _load_state_file(self, filepath: str) -> Set[str]:
        processed: Set[str] = set()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f: item = line.strip();
                if item: processed.add(item)
            logger.info(f"{len(processed)} items loaded from state file: {filepath}")
        except FileNotFoundError: logger.warning(f"State file not found: {filepath}")
        except Exception as e: logger.error(f"Error reading state file {filepath}: {e}", exc_info=True)
        return processed
    def _load_all_states(self):
        logger.info("Loading processing state...")
        self.google_searched_names = self._load_state_file(self.config.GOOGLE_SEARCHED_NAMES_FILE)
        self.google_search_no_results = self._load_state_file(self.config.GOOGLE_SEARCH_NO_RESULTS_FILE)
        self.processed_url_analysis = self._load_state_file(self.config.PROCESSED_URL_ANALYSIS_FILE)
        self.names_to_skip_completely = self.google_search_no_results.copy()
        logger.info("Processing state loaded.")
    def _add_to_state_file(self, filepath: str, item: str):
        try:
            with open(filepath, "a", encoding="utf-8") as f: f.write(item + "\n"); f.flush()
        except IOError as e: logger.error(f"Failed to write item '{item}' to state file {filepath}: {e}", exc_info=True)
    def add_google_searched_name(self, name: str):
        if name not in self.google_searched_names: self.google_searched_names.add(name); self._add_to_state_file(self.config.GOOGLE_SEARCHED_NAMES_FILE, name)
    def add_google_no_results_name(self, name: str):
        if name not in self.google_search_no_results: self.google_search_no_results.add(name); self.names_to_skip_completely.add(name); self._add_to_state_file(self.config.GOOGLE_SEARCH_NO_RESULTS_FILE, name)
    def add_processed_url_analysis(self, name: str, url: str):
        entry_key = f"{name}::{url}"
        if entry_key not in self.processed_url_analysis: self.processed_url_analysis.add(entry_key); self._add_to_state_file(self.config.PROCESSED_URL_ANALYSIS_FILE, entry_key)
    def should_skip_name(self, name: str) -> bool: return name in self.names_to_skip_completely
    def was_google_searched(self, name: str) -> bool: return name in self.google_searched_names
    def was_url_analyzed(self, name: str, url: str) -> bool: return f"{name}::{url}" in self.processed_url_analysis

# ===========================================
# Google Search Class (Unchanged)
# ===========================================
class GoogleSearcher:
    # (No changes needed here)
    def __init__(self, config: Config):
        self.config = config
        self.service: Optional[Resource] = self._initialize_service()
    def _initialize_service(self) -> Optional[Resource]:
        if not GOOGLE_API_AVAILABLE: return None
        if not self.config.GOOGLE_API_KEY or self.config.GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE" or \
           not self.config.GOOGLE_CX_ID or self.config.GOOGLE_CX_ID == "YOUR_CSE_ID_HERE":
            logger.error("Google API Key or CX ID is not configured correctly.")
            return None
        try:
            service: Resource = build("customsearch", "v1", developerKey=self.config.GOOGLE_API_KEY)
            logger.info("Google API client service created successfully.")
            return service
        except Exception as e: logger.critical(f"Failed to build Google API client service: {e}", exc_info=True); return None
    def search(self, query: str) -> List[Dict[str, str]]:
        if not self.service: return []
        results: List[Dict[str, str]] = []
        try:
            logger.info(f"Executing Google API query: '{query}'")
            res = self.service.cse().list(q=query, cx=self.config.GOOGLE_CX_ID, num=self.config.NUM_GOOGLE_RESULTS).execute()
            items = res.get("items", [])
            for item in items:
                link = item.get("link")
                if link: results.append({"link": link, "title": item.get("title", ""), "snippet": item.get("snippet", "")})
            logger.info(f"Google API returned {len(results)} results for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Google API search failed for query '{query}': {e}", exc_info=True)
            if 'quota' in str(e).lower(): logger.critical("Google API quota likely exceeded!")
            return []
    def close(self):
        if self.service and hasattr(self.service, 'close'):
            try: self.service.close(); logger.info("Google API client service closed.")
            except Exception as e: logger.warning(f"Error closing Google API service: {e}")

# ===========================================
# LLM Processor Class (Unchanged)
# ===========================================
class LLMProcessor:
    # (No changes needed here)
    def __init__(self, config: Config):
        self.config = config
        self.llm_filter: Optional[BaseChatModel] = None
        self.llm_analyzer: Optional[BaseChatModel] = None
        self.filter_prompt: Optional[ChatPromptTemplate] = None
        self.analyze_prompt: Optional[ChatPromptTemplate] = None
        self.json_parser: Optional[JsonOutputParser] = None
        self.string_parser: Optional[StrOutputParser] = None
        self._initialize_llms_and_prompts()
    def _initialize_llm(self, model_name: str) -> Optional[BaseChatModel]:
        if not LANGCHAIN_AVAILABLE or not self.config.GROQ_API_KEY: return None
        try:
            llm = ChatGroq(temperature=0.1, groq_api_key=self.config.GROQ_API_KEY, model_name=model_name, request_timeout=30, max_retries=2)
            logger.info(f"Groq LLM initialized successfully with model: {model_name}")
            return llm
        except Exception as e: logger.error(f"Failed to initialize Groq LLM ({model_name}): {e}", exc_info=True); return None
    def _initialize_llms_and_prompts(self):
        if not LANGCHAIN_AVAILABLE or not self.config.GROQ_API_KEY: logger.warning("Cannot initialize LLMs."); return
        self.llm_filter = self._initialize_llm(self.config.LLM_FILTER_MODEL_NAME)
        self.llm_analyzer = self._initialize_llm(self.config.LLM_ANALYZE_MODEL_NAME)
        if not self.llm_filter and not self.llm_analyzer: logger.error("Failed to initialize any LLM models."); return
        try:
            self.json_parser = JsonOutputParser()
            self.string_parser = StrOutputParser()
            self.filter_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research assistant specializing in contact information retrieval. Your goal is to analyze Google Search results (titles and snippets) and identify web pages MOST LIKELY to contain DIRECT contact information (email, phone number) or detailed professional profiles for a specific person. Prioritize official sites, contact pages, directories, professional profiles. De-prioritize news, general mentions, social media (unless contact info is in snippet). Respond ONLY in JSON format with a 'selected_links' key containing a list of promising URLs (empty list if none)."""),
                ("human", "Search results for '{person_name}':\n\n{search_results_text}\n\nSelect URLs most likely to contain direct contact or detailed profile info for '{person_name}'. Return JSON list under 'selected_links'.")])
            self.analyze_prompt = ChatPromptTemplate.from_messages([
                 ("system", """You are an expert information extractor. Extract profile details and contact info for '{person_name}' based ONLY on the text ('{page_text}') from URL '{source_url}'.
Instructions:
1. Analyze '{page_text}' for mentions of '{person_name}'.
2. Consider '{source_url}' context.
3. Extract: Current Job Title, Current Company, Location, Brief Summary/Bio (1-2 sentences), Direct Emails, Direct Phone numbers clearly associated with '{person_name}'.
4. IGNORE generic contacts unless explicitly linked. Use `null` or empty string/list if info not found/unclear.
5. Respond in STRICT JSON format with keys: "job_title", "company", "location", "summary", "associated_emails", "associated_phones". ONLY the JSON.
Example: {{"job_title": "Engineer", "company": "Tech", "location": "City, Country", "summary": "Bio here.", "associated_emails": ["a@b.com"], "associated_phones": ["123"]}} OR {{"job_title": null, ..., "associated_emails": [], ...}}"""),
                ("human", """Target: '{person_name}'
URL: {source_url}
Context:
---
{page_text}
---
Potential Contacts (for context only):
---
{contacts_found}
---
Extract profile and DIRECTLY associated contacts for '{person_name}'. Return JSON.""")])
            logger.info("LLM prompt templates and JSON parser created.")
        except Exception as e: logger.error(f"Failed to create LLM prompts or parser: {e}", exc_info=True); self.filter_prompt, self.analyze_prompt, self.json_parser, self.string_parser = None, None, None, None
    def filter_links(self, name: str, search_results: List[Dict[str, str]]) -> List[str]:
        if not self.llm_filter or not self.filter_prompt or not self.json_parser: logger.warning("LLM filter components unavailable. Returning all valid links."); return [r["link"] for r in search_results if r.get("link") and urlparse(r["link"]).scheme in ['http', 'https']]
        llm_name = getattr(self.llm_filter, 'model_name', 'N/A'); logger.info(f"Filtering {len(search_results)} Google results for '{name}' using LLM ({llm_name})...")
        formatted_results = "";
        for i, res in enumerate(search_results):
            link, title, snippet = res.get('link'), res.get('title'), res.get('snippet')
            if link and urlparse(link).scheme in ['http', 'https']: formatted_results += f"{i+1}. Title: {title or 'N/A'}\n   Snippet: {snippet or 'N/A'}\n   Link: {link}\n\n"
        if not formatted_results: logger.warning("No valid Google results for LLM filtering."); return []
        chain = self.filter_prompt | self.llm_filter | self.json_parser
        try:
            response = chain.invoke({"person_name": name, "search_results_text": formatted_results.strip()})
            if isinstance(response, dict) and "selected_links" in response:
                 selected = response["selected_links"]
                 if isinstance(selected, list): validated = [l for l in selected if isinstance(l, str) and urlparse(l).scheme in ['http', 'https']]; logger.info(f"LLM ({llm_name}) selected {len(validated)} relevant link(s)."); return validated
                 else: logger.warning(f"LLM ({llm_name}) filter 'selected_links' not a list: {response}"); return []
            else: logger.warning(f"Unexpected LLM ({llm_name}) filter response format: {response}"); return []
        except OutputParserException as ope: logger.error(f"LLM filter parsing error for '{name}': {ope}", exc_info=False); raw = getattr(ope, 'llm_output', str(ope)); logger.error(f"LLM ({llm_name}) raw output: {raw}"); return []
        except Exception as e: logger.error(f"LLM filter invocation error for '{name}': {e}", exc_info=True); return []
    def analyze_page_for_profile(self, name: str, url: str, page_content: str, all_emails: Set[str], all_phones: Set[str]) -> Dict[str, Any]:
        default_profile = {"job_title": None, "company": None, "location": None, "summary": None, "associated_emails": [], "associated_phones": []}
        if not self.llm_analyzer or not self.analyze_prompt or not self.json_parser or not self.string_parser: logger.warning("LLM analyzer components unavailable."); return default_profile
        if not page_content: logger.info("No page content to analyze."); return default_profile
        llm_name = getattr(self.llm_analyzer, 'model_name', 'N/A'); logger.info(f"Analyzing content from {url} for '{name}' profile using LLM ({llm_name})...")
        contacts_list_str = "Potential Emails Found:\n" + "\n".join(f"- {e}" for e in sorted(list(all_emails))) if all_emails else "Potential Emails Found: None"
        contacts_list_str += "\n\nPotential Phones Found:\n" + "\n".join(f"- {p}" for p in sorted(list(all_phones))) if all_phones else "\n\nPotential Phones Found: None"
        chain = self.analyze_prompt | self.llm_analyzer | self.json_parser
        fallback_chain = self.analyze_prompt | self.llm_analyzer | self.string_parser
        try:
            response = chain.invoke({"person_name": name, "source_url": url, "page_text": page_content, "contacts_found": contacts_list_str}, config={'max_retries': 1})
            if isinstance(response, dict):
                profile_data = {key: response.get(key, default_profile[key]) for key in default_profile};
                if not isinstance(profile_data["associated_emails"], list): profile_data["associated_emails"] = []
                if not isinstance(profile_data["associated_phones"], list): profile_data["associated_phones"] = []
                logger.info(f"LLM ({llm_name}) analysis complete for '{name}' on {url}.")
                return profile_data
            else: logger.warning(f"Unexpected LLM ({llm_name}) analysis response format (not dict): {response}"); return default_profile
        except OutputParserException as ope:
            logger.error(f"LLM analysis JSON parsing error for '{name}' on {url}: {ope}", exc_info=False)
            try: raw_output = fallback_chain.invoke({"person_name": name, "source_url": url, "page_text": page_content, "contacts_found": contacts_list_str}); logger.error(f"LLM ({llm_name}) raw output on parsing error: {raw_output}")
            except Exception as fallback_e: logger.error(f"Error invoking fallback string parser: {fallback_e}")
            return default_profile
        except Exception as e: logger.error(f"LLM analysis invocation error for '{name}' on {url}: {e}", exc_info=True); return default_profile

# ===========================================
# Web Scraper Class (Unchanged)
# ===========================================
class WebScraper:
    # (No changes needed here)
    def __init__(self, config: Config):
        self.config = config; self.playwright_manager: Optional[Playwright] = None; self.browser: Optional[Browser] = None; self.context: Optional[BrowserContext] = None; self.page: Optional[Page] = None; self._initialize()
    def _initialize(self):
        if not PLAYWRIGHT_AVAILABLE: logger.error("Playwright library not available."); return
        try:
            self.playwright_manager = sync_playwright().start(); logger.info("Launching headless Firefox browser...")
            self.browser = self.playwright_manager.firefox.launch(headless=True, args=["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage", "--no-sandbox"], slow_mo=random.uniform(50, 150))
            self.context = self.browser.new_context(user_agent=random.choice(self.config.USER_AGENTS), java_script_enabled=True, accept_downloads=False, ignore_https_errors=True)
            self.context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"); self.page = self.context.new_page(); logger.info("Playwright browser, context, and page initialized.")
        except Exception as e: logger.critical(f"Failed to initialize Playwright: {e}", exc_info=True); self.close()
    def extract_page_data(self, url: str) -> Tuple[Optional[str], Set[str], Set[str]]:
        if not self.page: logger.error("Playwright page object is not available."); return None, set(), set()
        all_emails: Set[str] = set(); all_phones: Set[str] = set(); page_text_content: Optional[str] = None
        try:
            logger.info(f"Navigating to: {url}"); response = self.page.goto(url, wait_until="domcontentloaded", timeout=self.config.PAGE_LOAD_TIMEOUT); status = response.status if response else 'N/A'; logger.info(f"Page loaded: {self.page.url} (Status: {status})")
            if response and not response.ok: logger.warning(f"Received non-OK status {status} for {url}")
            self.page.wait_for_timeout(random.randint(2500, 4000))
            page_text_content = self.page.evaluate("document.body.innerText")
            if page_text_content:
                original_length = len(page_text_content);
                if original_length > self.config.MAX_CONTENT_LENGTH: logger.warning(f"Content truncated from {original_length} to {self.config.MAX_CONTENT_LENGTH} chars."); page_text_content = page_text_content[:self.config.MAX_CONTENT_LENGTH]
                all_emails.update(self.config.EMAIL_REGEX.findall(page_text_content)); all_emails = {e.lower() for e in all_emails if '.' in e.split('@')[-1] and len(e.split('@')[0]) > 1 and not any(ext in e.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.js', '.css', '.webp', '.svg'])}
                found_phones_matches = self.config.PHONE_REGEX.findall(page_text_content);
                for phone_match in found_phones_matches: digits_only = re.sub(r'\D', '', phone_match);
                if 8 <= len(digits_only) <= 15: all_phones.add(phone_match.strip())
                try:
                    content_html = self.page.content(timeout=10000); mailto_links = re.findall(r'href=["\'](mailto:([^"\'?]+))["\']', content_html, re.IGNORECASE);
                    for _, email in mailto_links: email_lower = email.lower().strip();
                    if '@' in email_lower and '.' in email_lower.split('@')[-1]: all_emails.add(email_lower)
                    tel_links = re.findall(r'href=["\'](tel:([^"\'?]+))["\']', content_html, re.IGNORECASE);
                    for _, phone in tel_links: all_phones.add(phone.strip())
                except Exception as html_err: logger.warning(f"Error extracting mailto/tel links from HTML: {html_err}")
                logger.info(f"Extracted: {len(all_emails)} potential emails, {len(all_phones)} potential phones from {url}.")
            else: logger.warning(f"Could not extract text content from {url}.")
        except PlaywrightTimeoutError as pte: logger.error(f"Timeout error accessing {url}: {pte}")
        except PlaywrightError as pe: logger.error(f"Playwright error accessing {url}: {pe}")
        except Exception as e: logger.error(f"Unexpected error during page data extraction for {url}: {e}", exc_info=True)
        return page_text_content, all_emails, all_phones
    def close(self):
        if not self.playwright_manager and not self.browser: return; logger.info("Closing Playwright resources...")
        try:
            if self.page: self.page.close();
            if self.context: self.context.close();
            if self.browser: self.browser.close();
            if self.playwright_manager: self.playwright_manager.stop(); logger.info("Playwright closed successfully.")
        except Exception as e: logger.error(f"Error during Playwright cleanup: {e}", exc_info=True)
        finally: self.playwright_manager, self.browser, self.context, self.page = None, None, None, None

# ===========================================
# Main Orchestrator Class (Updated for Task Status)
# ===========================================
class ContactFinder:
    """Orchestrates the process of finding contacts and profiles for names."""
    def __init__(self, config: Config, task_id: str, input_data: List[str] | str):
        """
        Initializes the finder.
        Args:
            config: Configuration object.
            task_id: The unique ID for this processing task.
            input_data: Either a list of names (from text input) or a path to a CSV file.
        """
        self.config = config
        self.task_id = task_id
        self.input_data = input_data
        self.is_file_input = isinstance(input_data, str) and os.path.exists(input_data)
        self.input_source_name = os.path.basename(input_data) if self.is_file_input else f"text_input_{task_id}"
        self.output_file_path = self.config.get_output_filename(task_id) # Unique output per task

        self.state_manager = StateManager(config)
        self.google_searcher = GoogleSearcher(config)
        self.llm_processor = LLMProcessor(config)
        self.web_scraper: Optional[WebScraper] = None
        self.urls_to_analyze: Dict[str, List[str]] = {}

    def _update_status(self, status: str, message: str, error_detail: Optional[str] = None):
        """Helper to update task status via the global function."""
        update_task_status(
            self.task_id,
            status=status,
            message=message,
            output_file=self.output_file_path if status == 'COMPLETED' else None,
            log_file=LOG_FILE,
            error_detail=error_detail
        )

    def _read_input_names(self) -> Optional[List[str]]:
        """Reads unique names from the specified input source."""
        self._update_status('PROCESSING', f"Reading input: {self.input_source_name}")
        unique_names: List[str] = []
        try:
            if self.is_file_input and isinstance(self.input_data, str):
                input_df = pd.read_csv(self.input_data)
                if "nom" not in input_df.columns:
                    err_msg = f"Input file '{self.input_source_name}' missing required 'nom' column."
                    logger.critical(err_msg)
                    self._update_status('FAILED', "Input Error", error_detail=err_msg)
                    return None
                input_df.dropna(subset=['nom'], inplace=True)
                input_df['nom'] = input_df['nom'].astype(str).str.strip()
                input_df = input_df[input_df['nom'] != '']
                unique_names = input_df['nom'].unique().tolist()
            elif isinstance(self.input_data, list):
                # Input is already a list of names (from text area)
                unique_names = list(set(name.strip() for name in self.input_data if name.strip()))
            else:
                err_msg = "Invalid input data type provided."
                logger.critical(err_msg)
                self._update_status('FAILED', "Internal Error", error_detail=err_msg)
                return None

            if not unique_names:
                msg = f"No valid names found in input: {self.input_source_name}"
                logger.warning(msg)
                # Consider if this is a failure or just an empty completion
                self._update_status('COMPLETED', msg) # Or FAILED? Depends on desired behavior
                return None

            logger.info(f"Read {len(unique_names)} unique names from '{self.input_source_name}'.")
            self._update_status('PROCESSING', f"Read {len(unique_names)} unique names.")
            return unique_names
        except FileNotFoundError:
            err_msg = f"Input file '{self.input_source_name}' not found."
            logger.critical(err_msg)
            self._update_status('FAILED', "Input Error", error_detail=err_msg)
            return None
        except Exception as e:
            err_msg = f"Error reading input '{self.input_source_name}': {e}"
            logger.critical(err_msg, exc_info=True)
            self._update_status('FAILED', "Input Error", error_detail=err_msg)
            return None

    def _prepare_output_file(self):
        """Prepares the unique output file for this task."""
        self._update_status('PROCESSING', f"Preparing output file: {self.output_file_path}")
        output_file_exists = os.path.exists(self.output_file_path)
        try:
            # Use 'w' to ensure a clean file for each task run
            with open(self.output_file_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=self.config.OUTPUT_CSV_SEPARATOR) # Use configured separator
                logger.info(f"Writing header to '{self.output_file_path}'")
                writer.writerow(self.config.CSV_HEADERS)
                f.flush()
        except IOError as e:
            err_msg = f"Cannot open/write header to '{self.output_file_path}': {e}"
            logger.critical(err_msg)
            self._update_status('FAILED', "Output Error", error_detail=err_msg)
            raise # Re-raise to stop the process

    def _perform_google_search_step(self, names: List[str]):
        logger.info(f"[{self.task_id}] ===== Starting Step 1: Google Search & URL Filtering =====")
        self._update_status('PROCESSING', "Starting Google Search phase...")
        names_processed_count = 0
        total_names_to_process = len(names)
        urls_found_count = 0

        for i, name in enumerate(names):
            progress_msg = f"Google Search: Processing '{name}' ({i+1}/{total_names_to_process})"
            self._update_status('PROCESSING', progress_msg)

            if self.state_manager.should_skip_name(name):
                logger.info(f"[{self.task_id}] Skipping Google Search for '{name}' (no results).")
                continue
            if self.state_manager.was_google_searched(name):
                logger.info(f"[{self.task_id}] Skipping Google Search for '{name}' (already searched).")
                # If already searched, we might need to load previous results if not persisted
                # For simplicity here, we just skip. A more robust system might reload.
                continue

            names_processed_count += 1
            logger.info(f"[{self.task_id}] --- Processing Name {names_processed_count}/{total_names_to_process} (Search): '{name}' ---")
            filtered_urls = []
            if self.google_searcher.service:
                time.sleep(random.uniform(1.2, 2.8))
                search_query = f'"{name}" contact OR profile OR about OR email OR phone'
                google_results = self.google_searcher.search(search_query)
                if google_results:
                    time.sleep(random.uniform(0.6, 1.6))
                    filtered_urls = self.llm_processor.filter_links(name, google_results)
                else:
                    logger.info(f"[{self.task_id}] No Google results found for '{name}'.")
            else:
                logger.warning(f"[{self.task_id}] Google service unavailable for '{name}'.")

            if filtered_urls:
                self.urls_to_analyze[name] = filtered_urls
                self.state_manager.add_google_searched_name(name)
                urls_found_count += len(filtered_urls)
            else:
                self.state_manager.add_google_no_results_name(name)

        msg = f"Google Search phase complete. Found {urls_found_count} potential URLs for {len(self.urls_to_analyze)} names."
        logger.info(f"[{self.task_id}] {msg}")
        self._update_status('PROCESSING', msg)
        logger.info(f"[{self.task_id}] ===== Finished Step 1: Google Search & URL Filtering =====")


    def _perform_website_analysis_step(self):
        """Runs website scraping and profile/contact analysis."""
        logger.info(f"[{self.task_id}] ===== Starting Step 2: Website Analysis & Profile/Contact Extraction =====")
        self._update_status('PROCESSING', "Starting Website Analysis phase...")

        if not self.web_scraper or not self.web_scraper.page:
            err_msg = "Playwright unavailable, skipping analysis."
            logger.error(f"[{self.task_id}] {err_msg}")
            self._update_status('FAILED', "Scraper Error", error_detail=err_msg)
            return # Cannot proceed

        analysis_tasks = []
        for name, urls in self.urls_to_analyze.items():
            for url in urls:
                 if not self.state_manager.was_url_analyzed(name, url):
                     analysis_tasks.append({'name': name, 'url': url})

        total_analysis_tasks = len(analysis_tasks)
        analysis_processed_count = 0
        profiles_saved_count = 0
        logger.info(f"[{self.task_id}] Total URL analysis tasks for this run: {total_analysis_tasks}")
        if total_analysis_tasks == 0:
            logger.info(f"[{self.task_id}] No new URLs to analyze.")
            self._update_status('PROCESSING', "No new URLs needed analysis.")
            return # Nothing more to do in this step

        try:
            # Open file once for appending
            with open(self.output_file_path, "a", newline='', encoding="utf-8") as f_output:
                csv_writer = csv.writer(f_output, delimiter=self.config.OUTPUT_CSV_SEPARATOR)

                for i, task in enumerate(analysis_tasks):
                    name, url = task['name'], task['url']
                    analysis_processed_count += 1
                    progress_msg = f"Analyzing URL {analysis_processed_count}/{total_analysis_tasks}: {url} for '{name}'"
                    logger.info(f"[{self.task_id}] {progress_msg}")
                    self._update_status('PROCESSING', progress_msg)

                    page_content, all_emails, all_phones = self.web_scraper.extract_page_data(url)
                    profile_data = self.llm_processor.analyze_page_for_profile(name, url, page_content, all_emails, all_phones)

                    # Prepare row data using PROFILE_FIELDS_MAPPING
                    row_data = [name, url]
                    has_data = False
                    for json_key, _ in self.config.PROFILE_FIELDS_MAPPING.items():
                        value = profile_data.get(json_key)
                        if isinstance(value, list):
                            cell_value = ", ".join(value) # Join lists with comma-space
                            if value: has_data = True
                        elif value is not None and str(value).strip(): # Check if value is not None or empty string
                            cell_value = str(value)
                            has_data = True
                        else:
                            cell_value = ""
                        row_data.append(cell_value)

                    analyzer_source = getattr(self.llm_processor.llm_analyzer, 'model_name', 'N/A') if self.llm_processor.llm_analyzer else "LLM_N/A"
                    row_data.append(analyzer_source)

                    if has_data:
                        csv_writer.writerow(row_data)
                        f_output.flush()
                        profiles_saved_count += 1
                        logger.info(f"[{self.task_id}] Saved extracted profile/contact data for {name} from {url}")
                    else:
                         logger.info(f"[{self.task_id}] No specific profile/contact data associated with {name} found on {url} after analysis.")

                    self.state_manager.add_processed_url_analysis(name, url)
                    sleep_time = random.uniform(2.0, 5.0) # Slightly reduced sleep
                    logger.info(f"[{self.task_id}] Pause for {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)

            msg = f"Website Analysis phase complete. Saved {profiles_saved_count} profiles/contacts."
            logger.info(f"[{self.task_id}] {msg}")
            self._update_status('PROCESSING', msg) # Intermediate status update

        except IOError as e:
            err_msg = f"Error writing results to '{self.output_file_path}': {e}"
            logger.critical(f"[{self.task_id}] {err_msg}", exc_info=True)
            self._update_status('FAILED', "Output Error", error_detail=err_msg)
        except Exception as e:
            err_msg = f"Unexpected error during website analysis loop: {e}"
            logger.critical(f"[{self.task_id}] {err_msg}", exc_info=True)
            self._update_status('FAILED', "Analysis Error", error_detail=err_msg)

        logger.info(f"[{self.task_id}] ===== Finished Step 2: Website Analysis & Profile/Contact Extraction =====")


    def run(self):
        """Executes the entire contact finding workflow."""
        logger.info(f"[{self.task_id}] ===== Starting Contact Finder Workflow for {self.input_source_name} =====")
        self._update_status('PROCESSING', "Workflow starting...")

        if not PANDAS_AVAILABLE:
            self._update_status('FAILED', "Dependency Error", error_detail="Pandas library not available.")
            return
        if not PLAYWRIGHT_AVAILABLE:
            self._update_status('FAILED', "Dependency Error", error_detail="Playwright library not available.")
            return

        try:
            self._prepare_output_file() # Raises exception on failure
            unique_names = self._read_input_names()
            if unique_names is None:
                # Status already set in _read_input_names (FAILED or COMPLETED if empty)
                logger.warning(f"[{self.task_id}] No valid names to process.")
                # Ensure status is not PROCESSING if we exit here
                if task_statuses.get(self.task_id, {}).get('status') == 'PROCESSING':
                     self._update_status('COMPLETED', "No valid names found in input.")
                return

            self._update_status('PROCESSING', "Initializing web scraper...")
            self.web_scraper = WebScraper(self.config)
            if not self.web_scraper or not self.web_scraper.page:
                 err_msg = "Web scraper initialization failed."
                 logger.critical(f"[{self.task_id}] {err_msg}")
                 self._update_status('FAILED', "Scraper Error", error_detail=err_msg)
                 if self.web_scraper: self.web_scraper.close() # Attempt cleanup
                 return

            self._perform_google_search_step(unique_names)
            self._perform_website_analysis_step()

            # Final status update (if no errors occurred during steps)
            if task_statuses.get(self.task_id, {}).get('status') == 'PROCESSING':
                final_msg = f"Processing complete for {self.input_source_name}."
                logger.info(f"[{self.task_id}] {final_msg}")
                self._update_status('COMPLETED', final_msg)

        except Exception as e:
            err_msg = f"Unhandled error in main workflow: {e}"
            logger.critical(f"[{self.task_id}] {err_msg}", exc_info=True)
            # Ensure status is set to FAILED if an unexpected error occurs
            if task_statuses.get(self.task_id, {}).get('status') != 'FAILED':
                 self._update_status('FAILED', "Workflow Error", error_detail=err_msg)
        finally:
            if self.web_scraper: self.web_scraper.close()
            if self.google_searcher: self.google_searcher.close()
            logger.info(f"[{self.task_id}] ===== Contact Finder Workflow Finished =====")
            # Optional: Clean up temporary input file if it was created from text
            if not self.is_file_input and isinstance(self.input_data, str) and os.path.exists(self.input_data):
                 try:
                     os.remove(self.input_data)
                     logger.info(f"[{self.task_id}] Removed temporary text input file: {self.input_data}")
                 except Exception as e:
                     logger.error(f"[{self.task_id}] Error removing temporary text input file {self.input_data}: {e}")


# ===========================================
# Background Task Function (Updated)
# ===========================================
def run_contact_finding_process(task_id: str, input_data: List[str] | str):
    """
    Function to be run in the background by FastAPI.
    Accepts task_id and either a list of names or a file path.
    """
    logger.info(f"Background task {task_id} started.")
    try:
        config = Config() # Use default config
        finder = ContactFinder(config, task_id, input_data)
        finder.run() # This method now handles status updates internally
        logger.info(f"Background task {task_id} finished.")
    except Exception as e:
        err_msg = f"Critical error during background processing for task {task_id}: {e}"
        logger.critical(err_msg, exc_info=True)
        # Ensure final status is FAILED if exception bubbles up here
        update_task_status(task_id, 'FAILED', "Background Task Error", error_detail=err_msg, log_file=LOG_FILE)
    finally:
        # Optional: Clean up the temporary *uploaded* file if it was one
        # Text input temp file cleanup is handled within ContactFinder.run()
        if isinstance(input_data, str) and os.path.exists(input_data) and input_data.startswith(config.TEMP_UPLOAD_DIR):
             try:
                 os.remove(input_data)
                 logger.info(f"[{task_id}] Removed temporary uploaded file: {input_data}")
             except Exception as e:
                 logger.error(f"[{task_id}] Error removing temporary uploaded file {input_data}: {e}")


# ===========================================
# FastAPI Application Setup
# ===========================================
app = FastAPI(title="Contact Finder API")

# --- CORS Middleware ---
origins = [
    "http://localhost:5173", # Default Vite dev server port
    "http://localhost:3000", # Common React dev port
    # Add production frontend URL if needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------

# ===========================================
# API Endpoints
# ===========================================

@app.post("/process-names/", status_code=202) # Use 202 Accepted for background tasks
async def process_names_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file containing names under a 'nom' column")
):
    """
    Accepts a CSV file with names, starts the background processing task,
    and returns a task ID for status polling.
    """
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type or filename. Please upload a CSV file.")

    config = Config()
    # Create a unique temporary file path for the upload
    temp_file_path = os.path.join(config.TEMP_UPLOAD_DIR, f"upload_{uuid.uuid4()}.csv")

    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded file saved temporarily to: {temp_file_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file {file.filename}: {e}", exc_info=True)
        # Clean up partial file if save failed
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
    finally:
        await file.close()

    # Generate Task ID and initialize status
    task_id = str(uuid.uuid4())
    task_statuses[task_id] = {
        "task_id": task_id,
        "status": "PENDING",
        "progress_message": "Task received, pending start.",
        "input_filename": file.filename, # Store original filename for reference
        "output_file": None,
        "log_file": LOG_FILE,
        "error_detail": None
    }
    logger.info(f"Task {task_id} created for file: {file.filename}")

    # Add the long-running task to the background, passing the task_id and the temp file path
    background_tasks.add_task(run_contact_finding_process, task_id, temp_file_path)
    logger.info(f"Background task added for task_id: {task_id}")

    return {
        "message": "File received. Processing started. Poll status using the task ID.",
        "task_id": task_id,
        # "output_file": config.get_output_filename(task_id), # Can predict output name
        # "log_file": LOG_FILE
    }

# --- Model for Text Input ---
class NamesInput(BaseModel):
    names: List[str]

@app.post("/process-names-text/", status_code=202)
async def process_names_text_endpoint(
    names_input: NamesInput,
    background_tasks: BackgroundTasks
):
    """
    Accepts a list of names, starts the background processing task,
    and returns a task ID for status polling.
    """
    if not names_input.names:
        raise HTTPException(status_code=400, detail="No names provided in the request.")

    config = Config()
    cleaned_names = [name.strip() for name in names_input.names if name.strip()]
    if not cleaned_names:
         raise HTTPException(status_code=400, detail="No valid (non-empty) names provided.")

    # Generate Task ID and initialize status
    task_id = str(uuid.uuid4())
    task_statuses[task_id] = {
        "task_id": task_id,
        "status": "PENDING",
        "progress_message": "Task received from text input, pending start.",
        "input_filename": f"text_input_{task_id}", # Placeholder name
        "output_file": None,
        "log_file": LOG_FILE,
        "error_detail": None
    }
    logger.info(f"Task {task_id} created for text input ({len(cleaned_names)} names).")

    # Add the long-running task, passing the task_id and the list of names
    background_tasks.add_task(run_contact_finding_process, task_id, cleaned_names)
    logger.info(f"Background task added for task_id: {task_id}")

    return {
        "message": "Names received. Processing started. Poll status using the task ID.",
        "task_id": task_id,
    }


@app.get("/task-status/{task_id}")
async def get_task_status_endpoint(task_id: str = Path(..., title="The ID of the task to get status for")):
    """
    Poll this endpoint with the task_id received from the POST request
    to get the current status of the processing task.
    """
    status_info = task_statuses.get(task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found.")

    # Return the current status object
    return status_info


@app.get("/")
async def read_root():
    return {"message": "Contact Finder API is running. Use POST /process-names/ or POST /process-names-text/."}

# ===========================================
# Script Execution
# ===========================================
if __name__ == "__main__":
    logger.info("Starting FastAPI server with Uvicorn...")
    uvicorn.run("main_api:app", host="127.0.0.1", port=8000, log_level="info", reload=True) # Added reload=True for dev
