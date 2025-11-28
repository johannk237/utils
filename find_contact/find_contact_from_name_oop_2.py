# find_contact_from_name_oop.py
import time
import random
import os
import csv
import re
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import List, Dict, Set, Tuple, Optional, Any

# --- Dependency Checks & Imports ---
# Load environment variables first
load_dotenv()

# Configure Logging (before other imports that might log)
LOG_FILE = "main_processor_oop.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# Get a logger for this module
logger = logging.getLogger(__name__)

# --- Google API Imports ---
try:
    from googleapiclient.discovery import build, Resource
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    logger.warning("google-api-python-client not found. Google Search will be disabled.")
# -------------------------

# --- LangChain / Groq Imports ---
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser # Added StrOutputParser for safety
    from langchain_core.exceptions import OutputParserException
    from langchain_core.language_models.chat_models import BaseChatModel
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain or LangChain-Groq not found. LLM processing will be disabled.")
# --------------------------------

# --- Playwright Imports ---
try:
    from playwright.sync_api import (
        sync_playwright, Playwright, Browser, Page, BrowserContext,
        TimeoutError as PlaywrightTimeoutError,
        Error as PlaywrightError
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not found. Web scraping will be disabled.")
# -------------------------

# --- Pandas Import ---
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.critical("Pandas not found. Install with: pip install pandas. Exiting.")
    exit()
# --------------------


# ===========================================
# Configuration Class
# ===========================================
class Config:
    """Holds all configuration settings for the application."""
    # --- File Paths ---
    INPUT_NAMES_FILE: str = "noms.csv"
    FINAL_OUTPUT_FILE: str = "profiles_associated_by_name_final_oop.csv" # Updated output name
    # State Tracking
    GOOGLE_SEARCHED_NAMES_FILE: str = "processed_google_searched_names.txt"
    GOOGLE_SEARCH_NO_RESULTS_FILE: str = "processed_google_search_no_results.txt"
    PROCESSED_URL_ANALYSIS_FILE: str = "processed_url_profile_analysis.txt" # Updated state file name

    # --- API Keys & IDs (Loaded from .env) ---
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
    GOOGLE_CX_ID: Optional[str] = os.getenv("GOOGLE_CX_ID", "YOUR_CSE_ID_HERE")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

    # --- Google Search Settings ---
    NUM_GOOGLE_RESULTS: int = 10

    # --- LLM Settings ---
    LLM_FILTER_MODEL_NAME: str = os.getenv("LLM_FILTER_MODEL", "llama3-8b-8192")
    LLM_ANALYZE_MODEL_NAME: str = os.getenv("LLM_ANALYZE_MODEL", "llama3-70b-8192") # Powerful model needed for extraction

    # --- Playwright Settings ---
    USER_AGENTS: List[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    ]
    NAVIGATION_TIMEOUT: int = 35_000 # ms
    PAGE_LOAD_TIMEOUT: int = 45_000 # ms
    MAX_CONTENT_LENGTH: int = 25000 # Chars for LLM analysis context limit

    # --- Output Settings ---
    OUTPUT_CSV_SEPARATOR: str = ";"
    # Define the profile fields we want to extract and their corresponding CSV headers
    PROFILE_FIELDS_MAPPING: Dict[str, str] = {
        "job_title": "JobTitle",
        "company": "Company",
        "location": "Location",
        "summary": "Summary",
        "associated_emails": "AssociatedEmails",
        "associated_phones": "AssociatedPhones"
    }
    # Get the headers in the desired order
    CSV_HEADERS: List[str] = ["Nom", "WebsiteURL"] + list(PROFILE_FIELDS_MAPPING.values()) + ["LLMSource"]


    # --- Regular Expressions ---
    EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    PHONE_REGEX = re.compile(r'(?:(?:\+|00)\s*\d{1,3}[-.\s]?)?(?:\(\s*\d{1,4}\s*\)|(?:\d+))?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b')

    def __init__(self, input_names_file: Optional[str] = None):
        if input_names_file:
            self.INPUT_NAMES_FILE = input_names_file
        else:
             self.INPUT_NAMES_FILE = "noms.csv" # Default if not provided

        # Validate essential configurations
        if not self.GROQ_API_KEY:
            logger.warning("Groq API Key missing. LLM steps will be skipped.")
        if (not self.GOOGLE_API_KEY or self.GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE") or \
           (not self.GOOGLE_CX_ID or self.GOOGLE_CX_ID == "YOUR_CSE_ID_HERE"):
            logger.warning("Google API Key/CX ID missing. Google Search step will be skipped.")

# ===========================================
# State Management Class (Unchanged)
# ===========================================
class StateManager:
    """Handles loading and saving processing state to files."""
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
                for line in f:
                    item = line.strip()
                    if item:
                        processed.add(item)
            logger.info(f"{len(processed)} items loaded from state file: {filepath}")
        except FileNotFoundError:
            logger.warning(f"State file not found, starting fresh: {filepath}")
        except Exception as e:
            logger.error(f"Error reading state file {filepath}: {e}", exc_info=True)
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
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(item + "\n")
                f.flush()
        except IOError as e:
            logger.error(f"Failed to write item '{item}' to state file {filepath}: {e}", exc_info=True)

    def add_google_searched_name(self, name: str):
        if name not in self.google_searched_names:
            self.google_searched_names.add(name)
            self._add_to_state_file(self.config.GOOGLE_SEARCHED_NAMES_FILE, name)

    def add_google_no_results_name(self, name: str):
        if name not in self.google_search_no_results:
            self.google_search_no_results.add(name)
            self.names_to_skip_completely.add(name)
            self._add_to_state_file(self.config.GOOGLE_SEARCH_NO_RESULTS_FILE, name)

    def add_processed_url_analysis(self, name: str, url: str):
        entry_key = f"{name}::{url}"
        if entry_key not in self.processed_url_analysis:
            self.processed_url_analysis.add(entry_key)
            self._add_to_state_file(self.config.PROCESSED_URL_ANALYSIS_FILE, entry_key)

    def should_skip_name(self, name: str) -> bool:
        return name in self.names_to_skip_completely

    def was_google_searched(self, name: str) -> bool:
        return name in self.google_searched_names

    def was_url_analyzed(self, name: str, url: str) -> bool:
        return f"{name}::{url}" in self.processed_url_analysis

# ===========================================
# Google Search Class (Unchanged)
# ===========================================
class GoogleSearcher:
    """Handles Google Custom Search API interactions."""
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
        except Exception as e:
            logger.critical(f"Failed to build Google API client service: {e}", exc_info=True)
            return None

    def search(self, query: str) -> List[Dict[str, str]]:
        if not self.service: return []
        results: List[Dict[str, str]] = []
        try:
            logger.info(f"Executing Google API query: '{query}'")
            res = self.service.cse().list(q=query, cx=self.config.GOOGLE_CX_ID, num=self.config.NUM_GOOGLE_RESULTS).execute()
            items = res.get("items", [])
            for item in items:
                link = item.get("link")
                if link:
                    results.append({
                        "link": link,
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", "")
                    })
            logger.info(f"Google API returned {len(results)} results for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Google API search failed for query '{query}': {e}", exc_info=True)
            if 'quota' in str(e).lower(): logger.critical("Google API quota likely exceeded!")
            return []

    def close(self):
        if self.service and hasattr(self.service, 'close'):
            try:
                self.service.close()
                logger.info("Google API client service closed.")
            except Exception as e:
                logger.warning(f"Error closing Google API service: {e}")

# ===========================================
# LLM Processor Class (Prompts Updated)
# ===========================================
class LLMProcessor:
    """Handles LLM initialization and interactions for filtering and analysis."""
    def __init__(self, config: Config):
        self.config = config
        self.llm_filter: Optional[BaseChatModel] = None
        self.llm_analyzer: Optional[BaseChatModel] = None
        self.filter_prompt: Optional[ChatPromptTemplate] = None
        self.analyze_prompt: Optional[ChatPromptTemplate] = None
        # Use a safer parser combination for analysis
        self.json_parser: Optional[JsonOutputParser] = None
        self.string_parser: Optional[StrOutputParser] = None # Fallback parser
        self._initialize_llms_and_prompts()

    def _initialize_llm(self, model_name: str) -> Optional[BaseChatModel]:
        if not LANGCHAIN_AVAILABLE: return None
        if not self.config.GROQ_API_KEY: return None
        try:
            # Added timeout and retry settings for Groq API calls
            llm = ChatGroq(
                temperature=0.1,
                groq_api_key=self.config.GROQ_API_KEY,
                model_name=model_name,
                request_timeout=30, # Timeout in seconds
                max_retries=2
            )
            logger.info(f"Groq LLM initialized successfully with model: {model_name}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM ({model_name}): {e}", exc_info=True)
            return None

    def _initialize_llms_and_prompts(self):
        if not LANGCHAIN_AVAILABLE or not self.config.GROQ_API_KEY:
            logger.warning("Cannot initialize LLMs: Langchain/Groq unavailable or API key missing.")
            return

        self.llm_filter = self._initialize_llm(self.config.LLM_FILTER_MODEL_NAME)
        self.llm_analyzer = self._initialize_llm(self.config.LLM_ANALYZE_MODEL_NAME)

        if not self.llm_filter and not self.llm_analyzer:
            logger.error("Failed to initialize any LLM models.")
            return

        try:
            self.json_parser = JsonOutputParser()
            self.string_parser = StrOutputParser() # Initialize string parser

            # --- Filter Prompt (Refined) ---
            self.filter_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research assistant specializing in contact information retrieval.
Your goal is to analyze Google Search results (titles and snippets) and identify web pages MOST LIKELY to contain DIRECT contact information (email, phone number) or detailed professional profiles for a specific person.
Prioritize:
- Official personal or professional websites (e.g., personal blog, portfolio).
- Contact pages ('Contact Us', 'About Us' mentioning the person).
- Staff directories or profile pages on company/organization sites.
- Reputable professional networking sites (e.g., LinkedIn, specific industry directories) if context suggests it's the correct person.
- University faculty pages.
De-prioritize:
- News articles, general mentions, forums, social media (unless contact info is explicitly in snippet).
- Vague or unrelated results, product pages, general articles.
Respond ONLY in JSON format with a 'selected_links' key containing a list of the most promising URLs. If no results seem promising, return an empty list."""),
                ("human", """I searched for contact/profile information for '{person_name}'. Here are the Google search results:

{search_results_text}

Analyze these results. Select the URLs that are most likely to contain direct email/phone contact information OR detailed professional profile information (job title, company, location, bio) for '{person_name}'. Return your selection as a JSON list under the key 'selected_links'.""")
            ])

            # --- Analyze Prompt (Updated for Profile Extraction) ---
            self.analyze_prompt = ChatPromptTemplate.from_messages([
                 ("system", """You are an expert information extractor. Your goal is to extract specific profile details and contact information for '{person_name}' based ONLY on the provided web page text ('{page_text}') from the URL '{source_url}'.

Instructions:
1. Carefully read the '{page_text}' looking for information DIRECTLY related to '{person_name}'.
2. Consider the '{source_url}' for context (e.g., company site, personal blog).
3. Extract the following details if explicitly mentioned and clearly associated with '{person_name}':
    - Current Job Title (or primary role mentioned).
    - Current Company/Organization name.
    - Location (City, State/Region, Country if available).
    - A brief Summary or Biography (1-2 sentences max).
    - Direct Email addresses associated with the person.
    - Direct Phone numbers associated with the person.
4. IGNORE generic company contact info unless explicitly linked to '{person_name}'.
5. If a piece of information is not found or not clearly linked, use `null` or an empty string/list for that field.
6. Respond in STRICT JSON format with the following keys: "job_title", "company", "location", "summary", "associated_emails", "associated_phones".

Example JSON output:
{{
  "job_title": "Senior Software Engineer",
  "company": "Tech Innovations Inc.",
  "location": "San Francisco, CA, USA",
  "summary": "Experienced engineer specializing in cloud infrastructure and distributed systems.",
  "associated_emails": ["john.doe@techinnovations.com", "j.doe@personal.com"],
  "associated_phones": ["+1-415-555-1212", "555-987-6543 (Mobile)"]
}}
OR if nothing found:
{{
  "job_title": null,
  "company": null,
  "location": null,
  "summary": null,
  "associated_emails": [],
  "associated_phones": []
}}
"""),
                ("human", """Target Person: '{person_name}'
Source URL: {source_url}

Web Page Text Context:
---
{page_text}
---

Potential Contacts Found on Page (for cross-referencing context, do not assume association):
---
{contacts_found}
---

Extract the profile information (job title, company, location, summary) and any DIRECTLY associated contact details (emails, phones) for '{person_name}' based ONLY on the provided text context and source URL. Return the result in the specified JSON format.""")
            ])
            logger.info("LLM prompt templates and JSON parser created.")

        except Exception as e:
            logger.error(f"Failed to create LLM prompts or parser: {e}", exc_info=True)
            self.filter_prompt, self.analyze_prompt, self.json_parser, self.string_parser = None, None, None, None

    def filter_links(self, name: str, search_results: List[Dict[str, str]]) -> List[str]:
        """Uses the filter LLM to select relevant URLs."""
        if not self.llm_filter or not self.filter_prompt or not self.json_parser:
            logger.warning("LLM filter components unavailable. Returning all valid links.")
            return [result["link"] for result in search_results if result.get("link") and urlparse(result["link"]).scheme in ['http', 'https']]

        llm_name = getattr(self.llm_filter, 'model_name', 'N/A')
        logger.info(f"Filtering {len(search_results)} Google results for '{name}' using LLM ({llm_name})...")

        formatted_results = ""
        for i, res in enumerate(search_results):
            link, title, snippet = res.get('link'), res.get('title'), res.get('snippet')
            if link and urlparse(link).scheme in ['http', 'https']:
                formatted_results += f"{i+1}. Title: {title or 'N/A'}\n   Snippet: {snippet or 'N/A'}\n   Link: {link}\n\n"

        if not formatted_results:
            logger.warning("No valid Google results with links to format for LLM filtering.")
            return []

        # Use JsonOutputParser for filtering as it expects a list
        chain = self.filter_prompt | self.llm_filter | self.json_parser
        try:
            response = chain.invoke({
                "person_name": name,
                "search_results_text": formatted_results.strip()
            })
            if isinstance(response, dict) and "selected_links" in response:
                 selected_links_data = response["selected_links"]
                 if isinstance(selected_links_data, list):
                     validated_links = [link for link in selected_links_data if isinstance(link, str) and urlparse(link).scheme in ['http', 'https']]
                     logger.info(f"LLM ({llm_name}) selected {len(validated_links)} relevant link(s).")
                     return validated_links
                 else:
                     logger.warning(f"LLM ({llm_name}) filter response 'selected_links' is not a list: {response}")
                     return []
            else:
                logger.warning(f"Unexpected LLM ({llm_name}) filter response format (missing 'selected_links'): {response}")
                return []
        except OutputParserException as ope:
            logger.error(f"LLM filter response parsing error for '{name}': {ope}", exc_info=True)
            raw_output = getattr(ope, 'llm_output', str(ope))
            logger.error(f"LLM ({llm_name}) raw output: {raw_output}")
            return []
        except Exception as e:
            logger.error(f"LLM filter invocation error for '{name}': {e}", exc_info=True)
            return []

    # --- Updated analyze_page_for_profile method ---
    def analyze_page_for_profile(self, name: str, url: str, page_content: str, all_emails: Set[str], all_phones: Set[str]) -> Dict[str, Any]:
        """Uses the analysis LLM to extract profile fields and associate contacts."""
        # Default result structure
        default_profile = {
            "job_title": None, "company": None, "location": None, "summary": None,
            "associated_emails": [], "associated_phones": []
        }

        if not self.llm_analyzer or not self.analyze_prompt or not self.json_parser or not self.string_parser:
            logger.warning("LLM analyzer components unavailable. Cannot extract profile.")
            # Return default structure with potentially all contacts if LLM fails
            # default_profile["associated_emails"] = sorted(list(all_emails))
            # default_profile["associated_phones"] = sorted(list(all_phones))
            return default_profile

        if not page_content: # No need to call LLM if no content
            logger.info("No page content found to analyze with LLM.")
            return default_profile

        llm_name = getattr(self.llm_analyzer, 'model_name', 'N/A')
        logger.info(f"Analyzing content from {url} for '{name}' profile using LLM ({llm_name})...")

        # Include potential contacts in prompt for context, even if LLM should primarily use page_text
        contacts_list_str = "Potential Emails Found:\n" + "\n".join(f"- {e}" for e in sorted(list(all_emails))) if all_emails else "Potential Emails Found: None"
        contacts_list_str += "\n\nPotential Phones Found:\n" + "\n".join(f"- {p}" for p in sorted(list(all_phones))) if all_phones else "\n\nPotential Phones Found: None"

        # Use JsonOutputParser, but have StrOutputParser as fallback for debugging
        chain = self.analyze_prompt | self.llm_analyzer | self.json_parser
        # Fallback chain
        fallback_chain = self.analyze_prompt | self.llm_analyzer | self.string_parser

        try:
            response = chain.invoke({
                "person_name": name,
                "source_url": url,
                "page_text": page_content,
                "contacts_found": contacts_list_str
            }, config={'max_retries': 1})

            if isinstance(response, dict):
                # Validate and extract expected fields
                profile_data = {
                    "job_title": response.get("job_title"),
                    "company": response.get("company"),
                    "location": response.get("location"),
                    "summary": response.get("summary"),
                    "associated_emails": response.get("associated_emails", []),
                    "associated_phones": response.get("associated_phones", [])
                }
                # Ensure lists are lists
                if not isinstance(profile_data["associated_emails"], list): profile_data["associated_emails"] = []
                if not isinstance(profile_data["associated_phones"], list): profile_data["associated_phones"] = []

                logger.info(f"LLM ({llm_name}) analysis complete for '{name}' on {url}.")
                return profile_data
            else:
                logger.warning(f"Unexpected LLM ({llm_name}) analysis response format (not dict): {response}")
                return default_profile

        except OutputParserException as ope:
            logger.error(f"LLM analysis JSON parsing error for '{name}' on {url}: {ope}", exc_info=False) # Don't need full traceback for parsing error
            try:
                # Attempt to get raw string output for debugging
                raw_output = fallback_chain.invoke({
                    "person_name": name, "source_url": url, "page_text": page_content, "contacts_found": contacts_list_str
                })
                logger.error(f"LLM ({llm_name}) raw output on parsing error: {raw_output}")
            except Exception as fallback_e:
                logger.error(f"Error invoking fallback string parser: {fallback_e}")
            return default_profile
        except Exception as e:
            logger.error(f"LLM analysis invocation error for '{name}' on {url}: {e}", exc_info=True)
            return default_profile


# ===========================================
# Web Scraper Class (Unchanged)
# ===========================================
class WebScraper:
    """Handles Playwright browser interactions and page data extraction."""
    def __init__(self, config: Config):
        self.config = config
        self.playwright_manager: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._initialize()

    def _initialize(self):
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright library not available.")
            return
        try:
            self.playwright_manager = sync_playwright().start()
            logger.info("Launching headless Firefox browser...")
            self.browser = self.playwright_manager.firefox.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage", "--no-sandbox"],
                slow_mo=random.uniform(50, 150)
            )
            self.context = self.browser.new_context(
                user_agent=random.choice(self.config.USER_AGENTS),
                java_script_enabled=True,
                accept_downloads=False,
                ignore_https_errors=True,
            )
            self.context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.page = self.context.new_page()
            logger.info("Playwright browser, context, and page initialized.")
        except Exception as e:
            logger.critical(f"Failed to initialize Playwright: {e}", exc_info=True)
            self.close()

    def extract_page_data(self, url: str) -> Tuple[Optional[str], Set[str], Set[str]]:
        if not self.page:
            logger.error("Playwright page object is not available.")
            return None, set(), set()

        all_emails: Set[str] = set()
        all_phones: Set[str] = set()
        page_text_content: Optional[str] = None

        try:
            logger.info(f"Navigating to: {url}")
            response = self.page.goto(url, wait_until="domcontentloaded", timeout=self.config.PAGE_LOAD_TIMEOUT)
            status = response.status if response else 'N/A'
            logger.info(f"Page loaded: {self.page.url} (Status: {status})")

            if response and not response.ok:
                 logger.warning(f"Received non-OK status {status} for {url}")

            self.page.wait_for_timeout(random.randint(2500, 4000))

            page_text_content = self.page.evaluate("document.body.innerText")

            if page_text_content:
                original_length = len(page_text_content)
                if original_length > self.config.MAX_CONTENT_LENGTH:
                    logger.warning(f"Content truncated from {original_length} to {self.config.MAX_CONTENT_LENGTH} chars for LLM analysis.")
                    page_text_content = page_text_content[:self.config.MAX_CONTENT_LENGTH]

                # Find emails/phones in text content
                all_emails.update(self.config.EMAIL_REGEX.findall(page_text_content))
                all_emails = {e.lower() for e in all_emails if '.' in e.split('@')[-1] and len(e.split('@')[0]) > 1 and not any(ext in e.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.js', '.css', '.webp', '.svg'])}

                found_phones_matches = self.config.PHONE_REGEX.findall(page_text_content)
                for phone_match in found_phones_matches:
                    digits_only = re.sub(r'\D', '', phone_match)
                    if 8 <= len(digits_only) <= 15:
                         all_phones.add(phone_match.strip())

                # Find emails/phones in mailto/tel links
                try:
                    content_html = self.page.content(timeout=10000)
                    mailto_links = re.findall(r'href=["\'](mailto:([^"\'?]+))["\']', content_html, re.IGNORECASE)
                    for _, email in mailto_links:
                        email_lower = email.lower().strip()
                        if '@' in email_lower and '.' in email_lower.split('@')[-1]: all_emails.add(email_lower)

                    tel_links = re.findall(r'href=["\'](tel:([^"\'?]+))["\']', content_html, re.IGNORECASE)
                    for _, phone in tel_links: all_phones.add(phone.strip())
                except Exception as html_err:
                     logger.warning(f"Error extracting mailto/tel links from HTML: {html_err}")

                logger.info(f"Extracted: {len(all_emails)} potential emails, {len(all_phones)} potential phones from {url}.")
            else:
                logger.warning(f"Could not extract text content (document.body.innerText) from {url}.")

        except PlaywrightTimeoutError as pte:
            logger.error(f"Timeout error accessing {url}: {pte}")
        except PlaywrightError as pe:
            logger.error(f"Playwright error accessing {url}: {pe}")
        except Exception as e:
            logger.error(f"Unexpected error during page data extraction for {url}: {e}", exc_info=True)

        return page_text_content, all_emails, all_phones

    def close(self):
        if not self.playwright_manager and not self.browser: return
        logger.info("Closing Playwright resources...")
        try:
            if self.page: self.page.close()
            if self.context: self.context.close()
            if self.browser: self.browser.close()
            if self.playwright_manager: self.playwright_manager.stop()
            logger.info("Playwright closed successfully.")
        except Exception as e:
            logger.error(f"Error during Playwright cleanup: {e}", exc_info=True)
        finally:
            self.playwright_manager, self.browser, self.context, self.page = None, None, None, None

# ===========================================
# Main Orchestrator Class (Updated Analysis Step & Output)
# ===========================================
class ContactFinder:
    """Orchestrates the process of finding contacts and profiles for names."""
    def __init__(self, config: Config):
        self.config = config
        self.state_manager = StateManager(config)
        self.google_searcher = GoogleSearcher(config)
        self.llm_processor = LLMProcessor(config)
        self.web_scraper: Optional[WebScraper] = None
        self.urls_to_analyze: Dict[str, List[str]] = {}

    def _read_input_names(self) -> Optional[List[str]]:
        try:
            input_df = pd.read_csv(self.config.INPUT_NAMES_FILE)
            if "nom" not in input_df.columns:
                 logger.critical(f"Input file '{self.config.INPUT_NAMES_FILE}' missing required 'nom' column. Exiting.")
                 return None
            input_df.dropna(subset=['nom'], inplace=True)
            input_df['nom'] = input_df['nom'].astype(str).str.strip()
            input_df = input_df[input_df['nom'] != '']
            unique_names = input_df['nom'].unique().tolist()
            logger.info(f"Read {len(unique_names)} unique names from '{self.config.INPUT_NAMES_FILE}'.")
            return unique_names
        except FileNotFoundError:
            logger.critical(f"Input file '{self.config.INPUT_NAMES_FILE}' not found. Exiting.")
            return None
        except Exception as e:
            logger.critical(f"Error reading input file '{self.config.INPUT_NAMES_FILE}': {e}", exc_info=True)
            return None

    def _prepare_output_file(self):
        """Ensures the output file exists and has the correct header if new."""
        output_file_exists = os.path.exists(self.config.FINAL_OUTPUT_FILE)
        try:
            with open(self.config.FINAL_OUTPUT_FILE, "a", newline='', encoding="utf-8") as f_output_check:
                csv_writer_check = csv.writer(f_output_check)
                if not output_file_exists or os.path.getsize(self.config.FINAL_OUTPUT_FILE) == 0:
                    logger.info(f"Writing header to '{self.config.FINAL_OUTPUT_FILE}'")
                    # Use headers defined in Config
                    csv_writer_check.writerow(self.config.CSV_HEADERS)
                    f_output_check.flush()
        except IOError as e:
            logger.critical(f"Cannot open or write header to output file '{self.config.FINAL_OUTPUT_FILE}': {e}. Exiting.")
            raise

    def _perform_google_search_step(self, names: List[str]):
        """Runs the Google Search and filtering for all applicable names."""
        logger.info("===== Starting Step 1: Google Search & URL Filtering =====")
        names_processed_count = 0
        total_names_to_process = len(names)

        for name in names:
            if self.state_manager.should_skip_name(name):
                logger.info(f"Skipping Google Search for '{name}' (previously no results).")
                continue
            if self.state_manager.was_google_searched(name):
                logger.info(f"Skipping Google Search for '{name}' (already searched).")
                # Enhancement: Load previously found URLs if state persistence is added
                continue

            names_processed_count += 1
            logger.info(f"--- Processing Name {names_processed_count}/{total_names_to_process} (Search): '{name}' ---")

            filtered_urls = []
            if self.google_searcher.service:
                time.sleep(random.uniform(1.2, 2.8))
                # Slightly broader search query
                search_query = f'"{name}" contact OR profile OR about OR email OR phone'
                google_results = self.google_searcher.search(search_query)

                if google_results:
                    time.sleep(random.uniform(0.6, 1.6))
                    filtered_urls = self.llm_processor.filter_links(name, google_results)
                else:
                    logger.info(f"No Google results found for '{name}'.")
            else:
                 logger.warning(f"Google service unavailable, cannot search for '{name}'.")

            if filtered_urls:
                self.urls_to_analyze[name] = filtered_urls
                self.state_manager.add_google_searched_name(name)
            else:
                self.state_manager.add_google_no_results_name(name)

        logger.info("===== Finished Step 1: Google Search & URL Filtering =====")
        logger.info(f"Identified URLs for {len(self.urls_to_analyze)} names in this run.")

    # --- Updated Website Analysis Step ---
    def _perform_website_analysis_step(self):
        """Runs the website scraping and profile/contact analysis step."""
        logger.info("===== Starting Step 2: Website Analysis & Profile/Contact Extraction =====")
        if not self.web_scraper or not self.web_scraper.page:
            logger.error("Playwright page not available, skipping website analysis.")
            return

        analysis_tasks_count = sum(len(urls) for urls in self.urls_to_analyze.values())
        analysis_processed_count = 0
        logger.info(f"Total URL analysis tasks for this run: {analysis_tasks_count}")

        if analysis_tasks_count == 0:
            logger.info("No new URLs identified in Step 1 to analyze.")
            return

        try:
            with open(self.config.FINAL_OUTPUT_FILE, "a", newline='', encoding="utf-8") as f_output:
                csv_writer = csv.writer(f_output)

                for name, urls in self.urls_to_analyze.items():
                    logger.info(f"--- Analyzing URLs for '{name}' ---")
                    for url in urls:
                        entry_key = f"{name}::{url}"
                        if self.state_manager.was_url_analyzed(name, url):
                            logger.info(f"Skipping already analyzed: {url}")
                            continue

                        analysis_processed_count += 1
                        logger.info(f"Processing URL {analysis_processed_count}/{analysis_tasks_count}: {url}")

                        # Perform extraction and analysis
                        page_content, all_emails, all_phones = self.web_scraper.extract_page_data(url)
                        profile_data = self.llm_processor.analyze_page_for_profile(
                            name, url, page_content, all_emails, all_phones
                        )

                        # Prepare row data based on extracted profile_data
                        row_data = [name, url]
                        has_data = False # Flag to check if any profile/contact info was found
                        for json_key, csv_header in self.config.PROFILE_FIELDS_MAPPING.items():
                            value = profile_data.get(json_key)
                            if isinstance(value, list):
                                # Join list items (emails/phones)
                                cell_value = self.config.OUTPUT_CSV_SEPARATOR.join(value)
                                if value: has_data = True
                            elif value is not None:
                                # Use string value directly (job, company, etc.)
                                cell_value = str(value)
                                has_data = True
                            else:
                                cell_value = "" # Use empty string for null/None
                            row_data.append(cell_value)

                        # Add LLM source
                        analyzer_source = getattr(self.llm_processor.llm_analyzer, 'model_name', 'N/A') if self.llm_processor.llm_analyzer else "LLM_N/A"
                        row_data.append(analyzer_source)

                        # Write row only if some data was extracted by the LLM
                        if has_data:
                            csv_writer.writerow(row_data)
                            f_output.flush()
                            logger.info(f"Saved extracted profile/contact data for {name} from {url}")
                        else:
                             logger.info(f"No specific profile/contact data associated with {name} found on {url} after analysis.")

                        # Mark as processed regardless of finding data
                        self.state_manager.add_processed_url_analysis(name, url)

                        # Pause between URL analyses
                        sleep_time = random.uniform(4.0, 8.0)
                        logger.info(f"Pause for {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)

        except IOError as e:
            logger.critical(f"Error writing results to '{self.config.FINAL_OUTPUT_FILE}': {e}", exc_info=True)
        except Exception as e:
             logger.critical(f"Unexpected error during website analysis loop: {e}", exc_info=True)

        logger.info("===== Finished Step 2: Website Analysis & Profile/Contact Extraction =====")
    # --- End Updated Step ---

    def run(self):
        """Executes the entire contact finding workflow."""
        logger.info("===== Starting Contact Finder Workflow =====")

        if not PANDAS_AVAILABLE: return
        if not PLAYWRIGHT_AVAILABLE:
            logger.critical("Playwright is required for website analysis. Exiting.")
            return

        try:
            self._prepare_output_file()
            unique_names = self._read_input_names()
            if unique_names is None: return

            # Initialize WebScraper within a try/finally block for guaranteed cleanup
            self.web_scraper = WebScraper(self.config)
            if not self.web_scraper or not self.web_scraper.page:
                 logger.critical("Web scraper initialization failed. Exiting.")
                 # Ensure potential partial cleanup if init failed mid-way
                 if self.web_scraper: self.web_scraper.close()
                 return

            self._perform_google_search_step(unique_names)
            self._perform_website_analysis_step()

        except Exception as e:
            logger.critical(f"An unhandled error occurred in the main workflow: {e}", exc_info=True)
        finally:
            # Cleanup resources
            if self.web_scraper: self.web_scraper.close()
            if self.google_searcher: self.google_searcher.close()
            logger.info("===== Contact Finder Workflow Finished =====")


# ===========================================
# Entry Point
# ===========================================
if __name__ == "__main__":
    # --- Argument Parsing (Optional) ---
    import argparse
    parser = argparse.ArgumentParser(description="Find profile and contact information for names listed in a CSV file.")
    parser.add_argument(
        "-i", "--input",
        default=None, # Default to None, will use Config default if not provided
        help="Path to the input CSV file containing names (must have a 'nom' column)."
    )
    args = parser.parse_args()
    # ------------------------------------

    # Pass the input file path to the Config if provided via command line
    config = Config(input_names_file=args.input)
    finder = ContactFinder(config)
    finder.run()
