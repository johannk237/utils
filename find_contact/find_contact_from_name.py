# find_contact_from_name.py
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
LOG_FILE = "main_processor.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- Google API Imports ---
try:
    from googleapiclient.discovery import build, Resource
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    logging.warning("google-api-python-client not found. Google Search will be disabled.")
    logging.warning("Install with: pip install google-api-python-client")
# -------------------------

# --- LangChain / Groq Imports ---
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.exceptions import OutputParserException
    from langchain_core.language_models.chat_models import BaseChatModel # For type hinting
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain or LangChain-Groq not found. LLM filtering and analysis will be disabled.")
    logging.warning("Install with: pip install langchain langchain-groq python-dotenv")
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
    logging.warning("Playwright not found. Web scraping will be disabled.")
    logging.warning("Install with: pip install playwright && python -m playwright install")
# -------------------------

# --- Pandas Import ---
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Pandas is critical for reading input, so exit if not available
    logging.critical("Pandas not found. Install with: pip install pandas. Exiting.")
    exit()
# --------------------


# ===========================================
# Configuration Constants
# ===========================================
# --- File Paths ---*
INPUT_NAMES_FILE: str = "noms.csv"
FINAL_OUTPUT_FILE: str = "contacts_associated_by_name_final.csv"

# State Tracking
GOOGLE_SEARCHED_NAMES_FILE: str = "processed_google_searched_names.txt"
GOOGLE_SEARCH_NO_RESULTS_FILE: str = "processed_google_search_no_results.txt"
PROCESSED_URL_ANALYSIS_FILE: str = "processed_url_contact_analysis.txt"

# --- API Keys & IDs (Loaded from .env) ---
# Provide default placeholders if not in .env, but log errors if they are used.
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
GOOGLE_CX_ID: Optional[str] = os.getenv("GOOGLE_CX_ID", "YOUR_CSE_ID_HERE")
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

# --- Google Search Settings ---
NUM_GOOGLE_RESULTS: int = 10

# --- LLM Settings ---
# Recommended: Use faster/cheaper model for filtering, powerful model for analysis
LLM_FILTER_MODEL_NAME: str = os.getenv("LLM_FILTER_MODEL", "llama3-8b-8192")
LLM_ANALYZE_MODEL_NAME: str = os.getenv("LLM_ANALYZE_MODEL", "llama3-70b-8192")

# --- Playwright Settings ---
USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0", # Adjusted Firefox version
]
NAVIGATION_TIMEOUT: int = 35_000 # ms
PAGE_LOAD_TIMEOUT: int = 45_000 # ms
MAX_CONTENT_LENGTH: int = 25000 # Chars for LLM analysis context limit

# --- Output Settings ---
OUTPUT_CSV_SEPARATOR: str = ";" # Separator for multiple contacts in the final CSV

# --- Regular Expressions ---
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
# Improved phone regex - adjust country specifics if needed
PHONE_REGEX = re.compile(r'(?:(?:\+|00)\s*\d{1,3}[-.\s]?)?(?:\(\s*\d+\s*\)|(?:\d+))?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,5}\b')
# ----------------------------


# ===========================================
# State Management Functions
# ===========================================
def load_processed_set(filepath: str) -> Set[str]:
    """Loads a set of processed items from a file."""
    processed: Set[str] = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = line.strip()
                if item:
                    processed.add(item)
        logging.info(f"{len(processed)} items loaded from state file: {filepath}")
    except FileNotFoundError:
        logging.warning(f"State file not found, starting fresh: {filepath}")
    except Exception as e:
        logging.error(f"Error reading state file {filepath}: {e}", exc_info=True)
    return processed

def add_to_processed_file(filepath: str, item: str):
    """Appends an item to a state tracking file."""
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(item + "\n")
    except IOError as e:
        logging.error(f"Failed to write item '{item}' to state file {filepath}: {e}", exc_info=True)

# ===========================================
# Google Search Functions
# ===========================================
def initialize_google_service() -> Optional[Resource]:
    """Initializes the Google API client service."""
    if not GOOGLE_API_AVAILABLE:
        logging.error("Google API client library not available.")
        return None
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE" or \
       not GOOGLE_CX_ID or GOOGLE_CX_ID == "YOUR_CSE_ID_HERE":
        logging.error("Google API Key or CX ID is not configured correctly in .env or script.")
        return None
    try:
        service: Resource = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        logging.info("Google API client service created successfully.")
        return service
    except Exception as e:
        logging.critical(f"Failed to build Google API client service: {e}", exc_info=True)
        return None

def get_google_search_results(query: str, service: Resource, num_results: int) -> List[Dict[str, str]]:
    """Performs Google search, returns list of {'link', 'title', 'snippet'}."""
    if not service:
        logging.error("Google service not initialized, cannot perform search.")
        return []
    results: List[Dict[str, str]] = []
    try:
        logging.info(f"Executing Google API query: '{query}'")
        res = service.cse().list(q=query, cx=GOOGLE_CX_ID, num=num_results).execute()
        items = res.get("items", [])
        for item in items:
            link = item.get("link")
            if link: # Only include results with a link
                results.append({
                    "link": link,
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", "")
                })
        logging.info(f"Google API returned {len(results)} results for query: '{query}'")
        return results
    except Exception as e:
        logging.error(f"Google API search failed for query '{query}': {e}", exc_info=True)
        if 'quota' in str(e).lower():
            logging.critical("Google API quota likely exceeded!")
        return []

# ===========================================
# LLM Interaction Functions
# ===========================================
def initialize_llm(model_name: str) -> Optional[BaseChatModel]:
    """Initializes a Groq Chat LLM."""
    if not LANGCHAIN_AVAILABLE:
        logging.error("LangChain/Groq library not available.")
        return None
    if not GROQ_API_KEY:
        logging.error("GROQ_API_KEY not found in environment variables.")
        return None
    try:
        llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API_KEY, model_name=model_name)
        logging.info(f"Groq LLM initialized successfully with model: {model_name}")
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize Groq LLM ({model_name}): {e}", exc_info=True)
        return None

def create_llm_prompts_and_parser() -> Tuple[Optional[ChatPromptTemplate], Optional[ChatPromptTemplate], Optional[JsonOutputParser]]:
    """Creates the prompt templates and JSON parser for LLM interactions."""
    if not LANGCHAIN_AVAILABLE:
        return None, None, None

    try:
        json_parser = JsonOutputParser()

        filter_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert research assistant. Your goal is to identify web pages most likely to contain contact information (email, phone) for a specific person. Respond in JSON format with a 'selected_links' key containing a list of relevant URLs."),
            ("human", "I searched for '{person_name}'. Here are the Google search results:\n\n{search_results_text}\n\nAnalyze the titles and snippets. Select only the links most likely to contain contact details (email, phone) for '{person_name}'. Prioritize official sites, contact pages, directories, or professional profiles. Return an empty list if no links seem relevant. Provide the response in the requested JSON format.")
        ])

        analyze_prompt = ChatPromptTemplate.from_messages([
             ("system", """You are an expert contact information extractor. Your goal is to identify emails and phone numbers DIRECTLY associated with '{person_name}' based on the provided web page text ('{page_text}') and a list of potential contacts found ('{contacts_found}').

Instructions:
1. Carefully analyze '{page_text}' for mentions of '{person_name}'.
2. Examine the '{contacts_found}' list.
3. Based ONLY on the context in '{page_text}', determine which emails and phone numbers from '{contacts_found}' are directly linked to '{person_name}'. Ignore generic company contacts unless explicitly linked to the person.
4. Respond in STRICT JSON format with two keys:
   - "associated_emails": A list (possibly empty) of emails relevant to '{person_name}'.
   - "associated_phones": A list (possibly empty) of phone numbers relevant to '{person_name}'.
Return ONLY the JSON object.

Example JSON output:
{{
  "associated_emails": ["jane.doe@example.com"],
  "associated_phones": ["+1-555-123-4567"]
}}
"""),
            ("human", """Target Person: '{person_name}'

Web Page Text Context:
---
{page_text}
---

Potential Contacts Found on Page:
---
{contacts_found}
---

Analyze the context and potential contacts. Return the JSON containing ONLY the emails and phones directly associated with '{person_name}' based on the context.""")
        ])
        logging.info("LLM prompt templates and JSON parser created.")
        return filter_prompt, analyze_prompt, json_parser

    except Exception as e:
        logging.error(f"Failed to create LLM prompts or parser: {e}", exc_info=True)
        return None, None, None

def filter_links_with_llm(
    name: str,
    search_results: List[Dict[str, str]],
    llm: BaseChatModel,
    prompt_template: ChatPromptTemplate,
    output_parser: JsonOutputParser
) -> List[str]:
    """Uses LLM to filter relevant URLs from Google search results."""
    if not llm or not prompt_template or not output_parser:
        logging.warning("LLM components not available for filtering. Returning all valid links.")
        return [result["link"] for result in search_results if result.get("link")]

    logging.info(f"Filtering {len(search_results)} Google results for '{name}' using LLM ({getattr(llm, 'model_name', 'N/A')})...")

    formatted_results = ""
    for i, res in enumerate(search_results):
        link = res.get('link', 'N/A')
        title = res.get('title', 'N/A')
        snippet = res.get('snippet', 'N/A')
        # Ensure link is valid before adding
        if urlparse(link).scheme in ['http', 'https']:
            formatted_results += f"{i+1}. Title: {title}\n"
            formatted_results += f"   Snippet: {snippet}\n"
            formatted_results += f"   Link: {link}\n\n"

    if not formatted_results:
        logging.warning("No valid Google results with links to format for LLM filtering.")
        return []

    chain = prompt_template | llm | output_parser

    try:
        response = chain.invoke({
            "person_name": name,
            "search_results_text": formatted_results.strip()
        })

        if isinstance(response, dict) and "selected_links" in response:
             selected_links_data = response["selected_links"]
             if isinstance(selected_links_data, list):
                 # Basic validation of returned URLs
                 validated_links = [link for link in selected_links_data if isinstance(link, str) and urlparse(link).scheme in ['http', 'https']]
                 logging.info(f"LLM ({getattr(llm, 'model_name', 'N/A')}) selected {len(validated_links)} relevant link(s).")
                 return validated_links
             else:
                 logging.warning(f"LLM ({getattr(llm, 'model_name', 'N/A')}) filter response 'selected_links' is not a list: {response}")
                 return []
        else:
            logging.warning(f"Unexpected LLM ({getattr(llm, 'model_name', 'N/A')}) filter response format: {response}")
            return []

    except OutputParserException as ope:
        logging.error(f"LLM filter response parsing error for '{name}': {ope}", exc_info=True)
        raw_output = getattr(ope, 'llm_output', str(ope))
        logging.error(f"LLM ({getattr(llm, 'model_name', 'N/A')}) raw output: {raw_output}")
        return []
    except Exception as e:
        logging.error(f"LLM filter invocation error for '{name}': {e}", exc_info=True)
        return []

def analyze_contacts_with_llm(
    name: str,
    page_content: str,
    all_emails: Set[str],
    all_phones: Set[str],
    llm: BaseChatModel,
    prompt_template: ChatPromptTemplate,
    output_parser: JsonOutputParser
) -> Tuple[List[str], List[str]]:
    """Uses LLM to associate extracted contacts with the target name based on page content."""
    if not llm or not prompt_template or not output_parser:
        logging.warning("LLM components not available for analysis. Returning all extracted contacts.")
        return sorted(list(all_emails)), sorted(list(all_phones))

    if not page_content or not (all_emails or all_phones):
        logging.info("No page content or initial contacts found to analyze with LLM.")
        return [], []

    logging.info(f"Analyzing contacts for '{name}' using LLM ({getattr(llm, 'model_name', 'N/A')})...")

    contacts_list_str = "Emails found on page:\n" + "\n".join(f"- {e}" for e in sorted(list(all_emails))) if all_emails else "Emails found on page: None"
    contacts_list_str += "\n\nPhones found on page:\n" + "\n".join(f"- {p}" for p in sorted(list(all_phones))) if all_phones else "\n\nPhones found on page: None"

    chain = prompt_template | llm | output_parser

    try:
        response = chain.invoke({
            "person_name": name,
            "page_text": page_content,
            "contacts_found": contacts_list_str
        }, config={'max_retries': 1}) # Limit retries for analysis

        if isinstance(response, dict):
            associated_emails = response.get("associated_emails", [])
            associated_phones = response.get("associated_phones", [])
            if not isinstance(associated_emails, list): associated_emails = []
            if not isinstance(associated_phones, list): associated_phones = []
            logging.info(f"LLM ({getattr(llm, 'model_name', 'N/A')}) associated {len(associated_emails)} email(s) and {len(associated_phones)} phone(s) with '{name}'.")
        else:
            logging.warning(f"Unexpected LLM ({getattr(llm, 'model_name', 'N/A')}) analysis response format: {response}")
            associated_emails, associated_phones = [], []

    except OutputParserException as ope:
        logging.error(f"LLM analysis response parsing error for '{name}': {ope}", exc_info=True)
        raw_output = getattr(ope, 'llm_output', str(ope))
        logging.error(f"LLM ({getattr(llm, 'model_name', 'N/A')}) raw output: {raw_output}")
        associated_emails, associated_phones = [], []
    except Exception as e:
        logging.error(f"LLM analysis invocation error for '{name}': {e}", exc_info=True)
        associated_emails, associated_phones = [], []

    return sorted(associated_emails), sorted(associated_phones)

# ===========================================
# Playwright Web Scraping Functions
# ===========================================
def initialize_playwright() -> Optional[Dict[str, Any]]:
    """Initializes Playwright, browser, context, and page."""
    if not PLAYWRIGHT_AVAILABLE:
        logging.error("Playwright library not available.")
        return None
    try:
        p_manager = sync_playwright().start()
        logging.info("Launching headless Firefox browser...")
        browser = p_manager.firefox.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage", "--no-sandbox"],
            slow_mo=random.uniform(50, 150) # Slightly increased random slow_mo
        )
        context = browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            java_script_enabled=True,
            accept_downloads=False,
            ignore_https_errors=True, # Be cautious with this in production
        )
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()
        logging.info("Playwright browser, context, and page initialized.")
        return {"manager": p_manager, "browser": browser, "context": context, "page": page}
    except Exception as e:
        logging.critical(f"Failed to initialize Playwright: {e}", exc_info=True)
        # Attempt cleanup if partial initialization occurred
        if 'browser' in locals() and browser: browser.close()
        if 'p_manager' in locals() and p_manager: p_manager.stop()
        return None

def close_playwright(pw_context: Optional[Dict[str, Any]]):
    """Safely closes Playwright resources."""
    if not pw_context:
        return
    logging.info("Closing Playwright resources...")
    try:
        if pw_context.get("page"): pw_context["page"].close()
        if pw_context.get("context"): pw_context["context"].close()
        if pw_context.get("browser"): pw_context["browser"].close()
        if pw_context.get("manager"): pw_context["manager"].stop()
        logging.info("Playwright closed successfully.")
    except Exception as e:
        logging.error(f"Error during Playwright cleanup: {e}", exc_info=True)

def extract_page_data(page: Page, url: str) -> Tuple[Optional[str], Set[str], Set[str]]:
    """Navigates to URL, extracts text content and potential contacts."""
    if not page:
        logging.error("Playwright page object is not available.")
        return None, set(), set()

    all_emails: Set[str] = set()
    all_phones: Set[str] = set()
    page_text_content: Optional[str] = None

    try:
        logging.info(f"Navigating to: {url}")
        response = page.goto(url, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT)
        status = response.status if response else 'N/A'
        logging.info(f"Page loaded: {page.url} (Status: {status})")

        if response and not response.ok:
             logging.warning(f"Received non-OK status {status} for {url}")
             # Optionally decide whether to proceed based on status code

        page.wait_for_timeout(random.randint(2500, 4000)) # Increased random wait

        # Extract visible text using innerText
        page_text_content = page.evaluate("document.body.innerText")

        if page_text_content:
            original_length = len(page_text_content)
            if original_length > MAX_CONTENT_LENGTH:
                logging.warning(f"Content truncated from {original_length} to {MAX_CONTENT_LENGTH} chars for LLM analysis.")
                page_text_content = page_text_content[:MAX_CONTENT_LENGTH]

            # Find emails/phones in text content
            all_emails.update(EMAIL_REGEX.findall(page_text_content))
            # Further clean emails (basic validation)
            all_emails = {e.lower() for e in all_emails if '.' in e.split('@')[-1] and len(e.split('@')[0]) > 1 and not any(ext in e.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.js', '.css', '.webp', '.svg'])}

            found_phones_matches = PHONE_REGEX.findall(page_text_content)
            for phone_match in found_phones_matches:
                # Basic validation based on digits - adjust as needed
                digits_only = re.sub(r'\D', '', phone_match)
                if 8 <= len(digits_only) <= 15:
                     all_phones.add(phone_match.strip())

            # Find emails/phones in mailto/tel links (more reliable)
            try:
                content_html = page.content(timeout=10000)
                mailto_links = re.findall(r'href=["\'](mailto:([^"\'?]+))["\']', content_html, re.IGNORECASE)
                for _, email in mailto_links:
                    email_lower = email.lower().strip()
                    if '@' in email_lower and '.' in email_lower.split('@')[-1]:
                        all_emails.add(email_lower)

                tel_links = re.findall(r'href=["\'](tel:([^"\'?]+))["\']', content_html, re.IGNORECASE)
                for _, phone in tel_links:
                    # Keep original format from tel: link after stripping
                    all_phones.add(phone.strip())
            except Exception as html_err:
                 logging.warning(f"Error extracting mailto/tel links from HTML: {html_err}")

            logging.info(f"Extracted: {len(all_emails)} potential emails, {len(all_phones)} potential phones from {url}.")
        else:
            logging.warning(f"Could not extract text content (document.body.innerText) from {url}.")

    except PlaywrightTimeoutError as pte:
        logging.error(f"Timeout error accessing {url}: {pte}")
    except PlaywrightError as pe:
        logging.error(f"Playwright error accessing {url}: {pe}")
    except Exception as e:
        logging.error(f"Unexpected error during page data extraction for {url}: {e}", exc_info=True)

    return page_text_content, all_emails, all_phones

# ===========================================
# Main Orchestration Logic
# ===========================================
def _perform_google_search_and_filter(
    name: str,
    google_service: Optional[Resource],
    llm_filter: Optional[BaseChatModel],
    llm_filter_prompt: Optional[ChatPromptTemplate],
    json_parser: Optional[JsonOutputParser]
) -> List[str]:
    """Handles the Google Search and LLM filtering step for a single name."""
    if not google_service:
        logging.warning(f"Google service unavailable, cannot search for '{name}'.")
        return []

    logging.info(f"Performing Google Search for '{name}'...")
    time.sleep(random.uniform(1.2, 2.8)) # Pause before Google API
    search_query = f'"{name}"' # Simple query, rely on LLM filter
    google_results = get_google_search_results(search_query, google_service, NUM_GOOGLE_RESULTS)

    if not google_results:
        logging.info(f"No Google results found for '{name}'.")
        return []

    # Filter results using LLM
    if llm_filter and llm_filter_prompt and json_parser:
        time.sleep(random.uniform(0.6, 1.6)) # Pause before LLM filter API
        filtered_urls = filter_links_with_llm(name, google_results, llm_filter, llm_filter_prompt, json_parser)
    else:
        logging.warning("LLM filter unavailable, using all Google result links.")
        filtered_urls = [res["link"] for res in google_results if res.get("link")]

    # Validate URLs again after potential LLM filtering
    valid_filtered_urls = [url for url in filtered_urls if urlparse(url).scheme in ['http', 'https']]
    if len(valid_filtered_urls) < len(filtered_urls):
        logging.warning(f"Removed {len(filtered_urls) - len(valid_filtered_urls)} invalid URLs after filtering.")

    if valid_filtered_urls:
        logging.info(f"{len(valid_filtered_urls)} relevant URLs identified for '{name}'.")
    else:
        logging.info(f"No relevant URLs identified for '{name}' after filtering.")

    return valid_filtered_urls

def _perform_website_analysis(
    name: str,
    url: str,
    page: Optional[Page],
    llm_analyzer: Optional[BaseChatModel],
    llm_analyze_prompt: Optional[ChatPromptTemplate],
    json_parser: Optional[JsonOutputParser]
) -> Tuple[List[str], List[str]]:
    """Handles scraping a single URL and analyzing contacts using LLM."""
    if not page:
        logging.error(f"Playwright page not available, cannot analyze URL: {url}")
        return [], []

    logging.info(f"Analyzing URL: {url}")
    page_content, all_emails, all_phones = extract_page_data(page, url)

    associated_emails, associated_phones = [], []
    if page_content and (all_emails or all_phones):
        if llm_analyzer and llm_analyze_prompt and json_parser:
            time.sleep(random.uniform(0.7, 1.8)) # Pause before LLM analysis API
            associated_emails, associated_phones = analyze_contacts_with_llm(
                name, page_content, all_emails, all_phones, llm_analyzer, llm_analyze_prompt, json_parser
            )
        else:
            logging.warning("LLM analyzer unavailable, cannot associate contacts.")
            # Decide: return all found contacts or none? Returning none for stricter association.
            # associated_emails, associated_phones = sorted(list(all_emails)), sorted(list(all_phones))
    else:
        logging.info(f"No content or potential contacts extracted from {url} to analyze.")

    return associated_emails, associated_phones

def main():
    """Orchestrates the entire contact finding process."""
    logging.info("===== Starting Main Processor =====")

    # --- Dependency and API Key Checks ---
    if not PANDAS_AVAILABLE: return # Already logged critical error
    if (not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE") or \
       (not GOOGLE_CX_ID or GOOGLE_CX_ID == "YOUR_CSE_ID_HERE"):
        logging.warning("Google API Key/CX ID missing. Google Search step will be skipped.")
        google_service_active = False
    else:
        google_service_active = GOOGLE_API_AVAILABLE

    if not GROQ_API_KEY:
        logging.warning("Groq API Key missing. LLM steps will be skipped.")
        llm_active = False
    else:
        llm_active = LANGCHAIN_AVAILABLE

    if not PLAYWRIGHT_AVAILABLE:
        logging.warning("Playwright unavailable. Website analysis step will be skipped.")
    # ------------------------------------

    # --- Initialize Services ---
    google_service = initialize_google_service() if google_service_active else None
    llm_filter = initialize_llm(LLM_FILTER_MODEL_NAME) if llm_active else None
    llm_analyzer = initialize_llm(LLM_ANALYZE_MODEL_NAME) if llm_active else None
    llm_filter_prompt, llm_analyze_prompt, json_parser = create_llm_prompts_and_parser() if llm_active else (None, None, None)
    # -------------------------

    # --- Load State ---
    logging.info("Loading processing state...")
    google_searched_names = load_processed_set(GOOGLE_SEARCHED_NAMES_FILE)
    google_search_no_results = load_processed_set(GOOGLE_SEARCH_NO_RESULTS_FILE)
    processed_url_analysis = load_processed_set(PROCESSED_URL_ANALYSIS_FILE)
    names_to_skip_completely = google_search_no_results # Start with names known to have no results
    # ------------------

    # --- Read Input Names ---
    try:
        input_df = pd.read_csv(INPUT_NAMES_FILE)
        if "nom" not in input_df.columns:
             logging.critical(f"Input file '{INPUT_NAMES_FILE}' missing required 'nom' column. Exiting.")
             return
        input_df.dropna(subset=['nom'], inplace=True)
        input_df['nom'] = input_df['nom'].astype(str).str.strip()
        input_df = input_df[input_df['nom'] != '']
        unique_names = input_df['nom'].unique()
        logging.info(f"Read {len(unique_names)} unique names from '{INPUT_NAMES_FILE}'.")
    except FileNotFoundError:
        logging.critical(f"Input file '{INPUT_NAMES_FILE}' not found. Exiting.")
        return
    except Exception as e:
        logging.critical(f"Error reading input file '{INPUT_NAMES_FILE}': {e}", exc_info=True)
        return
    # -----------------------

    # --- Prepare Output File ---
    output_file_exists = os.path.exists(FINAL_OUTPUT_FILE)
    try:
        # Open output file once for writing headers if needed
        with open(FINAL_OUTPUT_FILE, "a", newline='', encoding="utf-8") as f_output_check:
            csv_writer_check = csv.writer(f_output_check)
            if not output_file_exists or os.path.getsize(FINAL_OUTPUT_FILE) == 0:
                logging.info(f"Writing header to '{FINAL_OUTPUT_FILE}'")
                csv_writer_check.writerow(["Nom", "WebsiteURL", "AssociatedEmails", "AssociatedPhones", "LLMSource"])
                f_output_check.flush()
    except IOError as e:
        logging.critical(f"Cannot open or write header to output file '{FINAL_OUTPUT_FILE}': {e}. Exiting.")
        return
    # -------------------------

    # --- Initialize Playwright ---
    # Use a context manager for Playwright resources
    pw_context_manager = None
    playwright_page: Optional[Page] = None
    if PLAYWRIGHT_AVAILABLE:
        pw_context_manager = initialize_playwright()
        if pw_context_manager:
            playwright_page = pw_context_manager.get("page")
        else:
            logging.error("Playwright initialization failed, web scraping disabled.")
            # Continue without Playwright if possible, or exit if critical
            # For this script, Playwright is essential for step 2.
            logging.critical("Playwright is essential for website analysis. Exiting.")
            return
    # ---------------------------

    # --- Main Processing Loop ---
    total_names_to_process = len(unique_names)
    names_processed_count = 0
    urls_to_analyze: Dict[str, List[str]] = {} # Store {name: [urls]} found in step 1

    # --- Step 1: Google Search and Filtering ---
    logging.info("===== Starting Step 1: Google Search & URL Filtering =====")
    for name in unique_names:
        if name in names_to_skip_completely:
            logging.info(f"Skipping Google Search for '{name}' (previously no results).")
            continue
        if name in google_searched_names:
            logging.info(f"Skipping Google Search for '{name}' (already searched).")
            # Need a way to retrieve previously found URLs if state isn't fully persistent
            # For now, we assume if it was searched, the URLs might be processed later if found
            continue

        names_processed_count += 1
        logging.info(f"--- Processing Name {names_processed_count}/{total_names_to_process} (Search): '{name}' ---")

        filtered_urls = _perform_google_search_and_filter(
            name, google_service, llm_filter, llm_filter_prompt, json_parser
        )

        if filtered_urls:
            urls_to_analyze[name] = filtered_urls
            add_to_processed_file(GOOGLE_SEARCHED_NAMES_FILE, name)
            google_searched_names.add(name) # Update in-memory set
        else:
            add_to_processed_file(GOOGLE_SEARCH_NO_RESULTS_FILE, name)
            names_to_skip_completely.add(name) # Update in-memory set

    logging.info("===== Finished Step 1: Google Search & URL Filtering =====")
    logging.info(f"Identified URLs for {len(urls_to_analyze)} names.")

    # --- Step 2: Website Analysis ---
    logging.info("===== Starting Step 2: Website Analysis & Contact Association =====")
    if not playwright_page:
        logging.error("Playwright page not available, skipping website analysis.")
    else:
        analysis_tasks_count = sum(len(urls) for urls in urls_to_analyze.values())
        analysis_processed_count = 0
        logging.info(f"Total URL analysis tasks: {analysis_tasks_count}")

        # Re-open output file in append mode for writing results
        try:
            with open(FINAL_OUTPUT_FILE, "a", newline='', encoding="utf-8") as f_output:
                csv_writer = csv.writer(f_output)

                for name, urls in urls_to_analyze.items():
                    logging.info(f"--- Analyzing URLs for '{name}' ---")
                    for url in urls:
                        analysis_processed_count += 1
                        entry_key = f"{name}::{url}"

                        if entry_key in processed_url_analysis:
                            logging.info(f"Skipping already analyzed: {url}")
                            continue

                        logging.info(f"Processing URL {analysis_processed_count}/{analysis_tasks_count}: {url}")

                        associated_emails, associated_phones = _perform_website_analysis(
                            name, url, playwright_page, llm_analyzer, llm_analyze_prompt, json_parser
                        )

                        # Write result row
                        email_str = OUTPUT_CSV_SEPARATOR.join(associated_emails)
                        phone_str = OUTPUT_CSV_SEPARATOR.join(associated_phones)
                        analyzer_source = getattr(llm_analyzer, 'model_name', 'N/A') if llm_analyzer else "LLM_N/A"
                        csv_writer.writerow([name, url, email_str, phone_str, analyzer_source])
                        f_output.flush()

                        # Mark as processed
                        add_to_processed_file(PROCESSED_URL_ANALYSIS_FILE, entry_key)
                        processed_url_analysis.add(entry_key) # Update in-memory set

                        # Pause between URL analyses
                        sleep_time = random.uniform(4.0, 8.0)
                        logging.info(f"Pause for {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)

        except IOError as e:
            logging.critical(f"Error writing results to '{FINAL_OUTPUT_FILE}': {e}", exc_info=True)
        except Exception as e:
             logging.critical(f"Unexpected error during website analysis loop: {e}", exc_info=True)


    logging.info("===== Finished Step 2: Website Analysis & Contact Association =====")
    # --------------------------

    # --- Final Cleanup ---
    close_playwright(pw_context_manager)
    if google_service and hasattr(google_service, 'close'):
        try:
            google_service.close()
            logging.info("Google API client service closed.")
        except Exception as e:
            logging.warning(f"Error closing Google API service: {e}")
    # ---------------------

    logging.info("===== Main Processor Finished =====")

# ===========================================
# Entry Point
# ===========================================
if __name__ == "__main__":
    # Check essential dependencies before running main
    if not PANDAS_AVAILABLE:
        logging.critical("Pandas is required but not installed. Please run: pip install pandas")
    else:
        main()
