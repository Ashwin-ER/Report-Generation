import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
import re
import io
import base64
from PIL import Image
import requests
from bs4 import BeautifulSoup
from collections import Counter

# Attempt to import duckduckgo_search
try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_SEARCH_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_SEARCH_AVAILABLE = False

# --- Custom NLP Utilities ---
CUSTOM_STOPWORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
    'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
    "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such',
    'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't",
    'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
    'also', 'however', 'therefore', 'thus', 'often', 'will', 'inc', 'llc', 'corp', 'ltd', 'gmbh', 'pvt', 'mr', 'mrs', 'ms', 'dr', 'eg', 'ie', 'etc', 'vs',
    'fig', 'figure', 'table', 'report', 'study', 'market', 'data', 'research', 'analysis', 'page', 'source', 'article',
    'company', 'companies', 'player', 'players', 'segment', 'segments', 'region', 'regions', 'growth', 'size', 'share',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
])

def custom_sent_tokenize(text):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def custom_word_tokenize(text):
    if not text:
        return []
    words = re.findall(r'\b\w+\b', text.lower())
    return words
# --- End Custom NLP Utilities ---

# Utility functions
def generate_placeholder_image(width=800, height=400, text="Placeholder Chart"):
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_bar_chart(title, x_label, y_label, labels, values):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_pie_chart(title, labels, values):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title(title)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_line_chart(title, x_label, y_label, data_dict):
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, values in data_dict.items():
        ax.plot(range(len(values)), values, label=key)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def get_image_download_link(img_buf, filename="chart.png"):
    img_str = base64.b64encode(img_buf.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">Download Chart</a>'
    return href

class WebResearcher:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.search_results_cache = {} # Simple instance-level cache
        self.page_content_cache = {}   # Simple instance-level cache
        
    def safe_request(self, url, retries=2, timeout=7): # Slightly reduced retries/timeout for cloud
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=timeout, allow_redirects=True)
                response.raise_for_status() # Will raise an HTTPError for bad responses (4XX or 5XX)
                return response
            except requests.exceptions.Timeout:
                if attempt == retries - 1:
                    st.sidebar.warning(f"Timeout fetching {url} after {retries} attempts.")
                    return None
                time.sleep(0.3)
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    st.sidebar.warning(f"Failed to fetch {url}: {str(e)[:100]}...")
                    return None
                time.sleep(0.3)
            except Exception as e: # Catch any other unexpected error during request
                st.sidebar.error(f"Unexpected error during request to {url}: {str(e)[:100]}...")
                return None
        return None
    
    def search_web_ddg(self, query, num_results=3):
        if not DUCKDUCKGO_SEARCH_AVAILABLE:
            st.sidebar.info("DuckDuckGo Search library not available. Using simulated search results.")
            return self.simulate_search_results(query, num_results)

        cache_key = f"ddg_{query}_{num_results}"
        if cache_key in self.search_results_cache:
            return self.search_results_cache[cache_key]

        try:
            results = []
            # DDGS can sometimes be slow or timeout, especially on free cloud platforms
            with DDGS(timeout=10) as ddgs: # Increased timeout for DDGS itself
                ddg_results = ddgs.text(query, max_results=num_results)
                for r in ddg_results:
                    results.append({
                        'title': r.get('title', 'No Title'),
                        'link': r.get('href', ''),
                        'snippet': r.get('body', '')
                    })
            if not results:
                st.sidebar.info(f"DuckDuckGo search for '{query}' yielded no results. Simulating.")
                return self.simulate_search_results(query, num_results)
            
            self.search_results_cache[cache_key] = results
            return results
        except Exception as e:
            st.sidebar.error(f"DuckDuckGo search error: {e}. Using simulated results.")
            return self.simulate_search_results(query, num_results)

    def get_content_from_url(self, url):
        if not url or not url.startswith(('http://', 'https://')):
             st.sidebar.warning(f"Skipping invalid URL: {url}")
             return ""
        if url in self.page_content_cache:
            return self.page_content_cache[url]

        response = self.safe_request(url)
        if response:
            try:
                # Ensure correct encoding if possible
                response.encoding = response.apparent_encoding if response.apparent_encoding else 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser') # Use response.text for BeautifulSoup

                for tag_name in ["script", "style", "nav", "footer", "aside", "header", "form", "button", "iframe", "noscript", "link", "meta"]:
                    for tag in soup.find_all(tag_name):
                        tag.decompose()
                
                text_parts = []
                main_content_selectors = [
                    'main', 'article', 
                    {'role': 'main'}, 
                    {'class': re.compile(r'(content|main|body|post|entry|article|story)', re.I)}
                ]
                content_container = None
                for selector in main_content_selectors:
                    if isinstance(selector, str):
                        content_container = soup.find(selector)
                    else:
                        content_container = soup.find('div', selector) # Common for class/role based
                    if content_container:
                        break
                
                if not content_container: # Fallback to body if no specific main content found
                    content_container = soup.body
                    if not content_container: # If even body is not found (very unlikely)
                        st.sidebar.warning(f"Could not find body tag in {url}")
                        return ""

                # More targeted extraction
                for element in content_container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'div', 'span'], recursive=True):
                    text = ""
                    if element.name == 'p' and len(element.get_text(strip=True)) > 40:
                        text = element.get_text(separator=" ", strip=True)
                    elif element.name in ['h1','h2','h3','h4']:
                        text = element.get_text(separator=" ", strip=True) + "."
                    elif element.name == 'li' and len(element.get_text(strip=True)) > 10:
                        text = "- " + element.get_text(separator=" ", strip=True)
                    elif element.name == 'div' and not element.find(['nav','aside','footer','header','form']) and len(element.get_text(strip=True)) > 80 :
                        # Avoid divs that are just containers for other complex non-textual elements
                        if not element.find(lambda tag: tag.name not in ['p','h1','h2','h3','h4','li','span','strong','em','a','br','img','figure','table','ul','ol'] and tag.string and tag.string.strip()):
                             text = element.get_text(separator=" ", strip=True)
                    elif element.name == 'span' and element.parent.name not in ['button','label','a','script','style'] and len(element.get_text(strip=True)) > 25:
                        text = element.get_text(separator=" ", strip=True)
                    
                    if text:
                        text_parts.append(text)

                if not text_parts: # If targeted tags yield nothing, try a broader get_text on container
                    broad_text = content_container.get_text(separator='\n', strip=True)
                    text_parts = [line for line in broad_text.split('\n') if line and len(line.strip()) > 20]

                text_content = "\n".join(text_parts)
                text_content = re.sub(r'\s*\n\s*[\n\s]*', '\n', text_content) 
                text_content = re.sub(r'[ \t]+', ' ', text_content)   
                text_content = text_content.strip()

                self.page_content_cache[url] = text_content
                return text_content
            except Exception as e:
                st.sidebar.warning(f"Error parsing content from {url}: {str(e)[:100]}...")
        return ""
        
    def research_topic_online(self, topic_query, is_industry_query):
        if is_industry_query:
            search_query_detail = "market analysis report size growth key players trends challenges forecast 2023 2024 2025"
        else:
            search_query_detail = "overview summary key points information facts"
        search_query = f"{topic_query} {search_query_detail}"

        st.write(f"🌐 Searching web for: '{search_query}'...")
        search_results_metadata = self.search_web_ddg(search_query, num_results=5) # Fetch more results to have fallbacks

        combined_text_for_processing = ""
        fetched_sources_list = []
        successful_fetches = 0
        attempted_fetches = 0
        max_successful_fetches = 2 # Limit successful full content fetches for speed

        if not search_results_metadata:
            st.error("🚫 No search results found online. Report will rely on any predefined data or be very limited.", icon="🛑")
            market_data_from_web = self.extract_market_data("", topic_query) if is_industry_query else {}
            return market_data_from_web, [], ""

        st.write(f"📑 Attempting to fetch detailed content from up to {len(search_results_metadata)} search results (will process up to {max_successful_fetches} successfully fetched pages):")
        
        for i, result in enumerate(search_results_metadata):
            if successful_fetches >= max_successful_fetches:
                st.write(f"ℹ️ Reached target of {max_successful_fetches} successfully fetched pages.")
                break
            
            link = result.get('link')
            title = result.get('title', 'No Title')
            
            if link and link.startswith('http'):
                attempted_fetches +=1
                st.write(f"   Attempt {attempted_fetches}: Fetching '{title}' ({link})...")
                content = self.get_content_from_url(link)
                if content and len(content) > 250: # Stricter length for meaningful content
                    combined_text_for_processing += f"\n\n--- Source: {title} ({link}) ---\n\n" + content + "\n\n" 
                    fetched_sources_list.append({'title': title, 'link': link})
                    successful_fetches += 1
                    st.write(f"     ✅ Content successfully fetched and added from '{title}'. ({successful_fetches}/{max_successful_fetches})")
                    time.sleep(0.1) # Small delay
                else:
                    st.write(f"     ⚠️ Skipped or failed to fetch substantial content from '{title}'.")
            else:
                st.write(f"  Skipping result with no valid link: '{title}'")

        if successful_fetches == 0 and attempted_fetches > 0:
            st.error(f"⚠️ Could not fetch detailed content from any of the {attempted_fetches} primary online sources attempted. The report will be generated using search result snippets and/or predefined data. This may significantly limit its depth and accuracy.", icon="❗")
            # Populate with snippets if no full content
            for res in search_results_metadata:
                if res.get('snippet'):
                    combined_text_for_processing += f"\n\n--- Snippet from: {res.get('title')} ---\n\n" + res.get('snippet', '') + "\n\n"
                    if not any(s['link'] == res.get('link') for s in fetched_sources_list) and res.get('link'): # Add as source if not already
                        fetched_sources_list.append({'title': f"Snippet: {res.get('title', 'Source')}", 'link': res.get('link')})
        elif successful_fetches < max_successful_fetches and successful_fetches > 0 and attempted_fetches > successful_fetches:
             st.warning(f"ℹ️ Fetched detailed content from {successful_fetches} source(s). Some other attempts were not successful. Using available content and snippets if necessary.", icon="💡")
             # Add snippets from sources that were not successfully fetched IF snippets exist and are not already covered
             for res in search_results_metadata:
                 if res.get('snippet') and not any(s['link'] == res.get('link') for s in fetched_sources_list):
                     combined_text_for_processing += f"\n\n--- Snippet from: {res.get('title')} ---\n\n" + res.get('snippet', '') + "\n\n"
                     fetched_sources_list.append({'title': f"Snippet: {res.get('title', 'Source')}", 'link': res.get('link')})


        if not combined_text_for_processing.strip(): # Still no text (e.g. no snippets from DDG either)
            st.error("🚫 No text content could be gathered from web (neither full pages nor snippets). Report will be based solely on simulations or predefined data if applicable.", icon="🛑")
            sim_results = self.simulate_search_results(topic_query, 1) # Try to get at least one simulation
            if sim_results and sim_results[0].get('snippet'):
                combined_text_for_processing = sim_results[0]['snippet']
                fetched_sources_list.append({'title': sim_results[0]['title'], 'link': sim_results[0]['link']})

        market_data_from_web = {}
        if is_industry_query and combined_text_for_processing.strip(): # Only extract if there's text
            market_data_from_web = self.extract_market_data(combined_text_for_processing, topic_query) 
    
        return market_data_from_web, fetched_sources_list, combined_text_for_processing.strip()

    def extract_market_data(self, combined_text, industry_topic): 
        # This function remains largely the same as it's about parsing text, not fetching it.
        # Its effectiveness depends on the quality of combined_text.
        market_data = {
            "market_size": "", "cagr": "", "key_players": [], "trends": [], "challenges": [],
            "year": datetime.now().year, "base_year_market_size": "", "forecast_year_market_size": ""
        }
        if not combined_text: return market_data

        size_patterns = [
            r"(?:valued at|market size was)\s*(USD|EUR|GBP|¥|\$)\s*([\d,]+\.?\d*)\s*(billion|million|trillion)\s*(?:in|as of)\s*(\d{4}).*?(?:(?:reach|grow to|projected at)\s*(USD|EUR|GBP|¥|\$)\s*([\d,]+\.?\d*)\s*(billion|million|trillion)\s*(?:by|in)\s*(\d{4}))?",
            r"(USD|EUR|GBP|¥|\$)\s*([\d,]+\.?\d*)\s*(billion|million|trillion)\s*(?:market|industry)\s*(?:in|as of)?\s*(\d{4})?",
        ]
        found_size = False
        for pattern in size_patterns:
            for match in re.finditer(pattern, combined_text, re.IGNORECASE):
                base_currency, base_value, base_unit, base_year = match.group(1), match.group(2), match.group(3), match.group(4)
                market_data["market_size"] = f"{base_currency.replace('¥','$')}{base_value} {base_unit} ({base_year})"
                market_data["base_year_market_size"] = f"{base_currency.replace('¥','$')}{base_value} {base_unit} ({base_year})"
                
                if match.group(5) and match.group(6) and match.group(7) and match.group(8):
                    fc_currency, fc_value, fc_unit, fc_year = match.group(5), match.group(6), match.group(7), match.group(8)
                    market_data["forecast_year_market_size"] = f"{fc_currency.replace('¥','$')}{fc_value} {fc_unit} ({fc_year})"
                    market_data["market_size"] = f"{fc_currency.replace('¥','$')}{fc_value} {fc_unit} (Projected {fc_year})"
                found_size = True; break
            if found_size: break
        
        if not found_size:
            match = re.search(r"(\$[\d,]+\.?\d*\s*(?:billion|million|trillion))", combined_text, re.IGNORECASE)
            if match: market_data["market_size"] = match.group(1)

        cagr_patterns = [
            r"CAGR of\s*([\d.]+\s*%)", r"compound annual growth rate.*?\s*([\d.]+\s*%)",
            r"grow at\s*(?:a CAGR of)?\s*([\d.]+\s*%)", r"expected to grow from.*?at\s*([\d.]+\s*%)",
            r"at a rate of\s*([\d.]+\s*%)",
        ]
        for pattern in cagr_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match: market_data["cagr"] = match.group(1).replace(" ", ""); break
        
        companies = []
        player_intro_patterns = [
             r"(?:key players|major players|leading companies|prominent players|market leaders|significant players|key vendors|major vendors|top companies|dominant players)\s*(?:profiled|include|are|such as|comprise|operating in this market are|covered in the report are|analyzed in this report are)\s*:?\s*([^.]+)\.",
             r"The report profiles key players such as\s*(.*?)(?:and other prominent vendors|which include|among others|The study also includes|$)",
             r"Some of the major companies that are present in the market are\s*(.*?)\."
        ]
        raw_player_strings = [match_obj.group(1) for pattern in player_intro_patterns for match_obj in re.finditer(pattern, combined_text, re.IGNORECASE)]
        
        specific_company_patterns_map = {
            "electric vehicle": r"\b(Tesla|BYD|Volkswagen|SAIC|Stellantis|Mercedes-Benz|Ford|General Motors|Hyundai-Kia|Toyota|NIO|XPeng|Li Auto)\b",
            "artificial intelligence": r"\b(Google|Alphabet|Microsoft|Amazon|AWS|NVIDIA|IBM|Meta|OpenAI|Anthropic|Baidu|Apple|Salesforce|Oracle|Intel|AMD|Palantir|C3.ai)\b",
            "renewable energy": r"\b(NextEra Energy|Enel|Iberdrola|EDF|Ørsted|Vestas|First Solar|Canadian Solar|Siemens Gamesa|LONGi|Jinko Solar|Trina Solar|GE Renewable Energy)\b",
            "cloud computing": r"\b(Amazon Web Services|AWS|Microsoft Azure|Azure|Google Cloud Platform|GCP|Alibaba Cloud|IBM Cloud|Oracle Cloud|Salesforce|SAP|VMware|Rackspace|DigitalOcean)\b"
        }
        generic_company_pattern = r"\b([A-Z][\w\s&-]+(?:Inc\.|LLC|Corp\.|Ltd\.|GmbH|S\.A\.))\b"
        companies.extend(re.findall(generic_company_pattern, combined_text))

        for industry_keyword, pattern in specific_company_patterns_map.items():
            if industry_keyword in industry_topic.lower():
                companies.extend(m for m in re.findall(pattern, combined_text) if m and m not in companies)

        for player_list_str in raw_player_strings:
            potential_players = re.split(r',\s*(?:and\s+)?|\s+and\s+|;\s*|\s*&\s*', player_list_str)
            for player in potential_players:
                player = player.strip()
                if player and len(player.split()) <= 5 and player[0].isupper() and not player.lower().endswith(("etc.", "e.g.", "others")) and not player.lower() in ["various", "many", "several", "leading", "major", "key"] and 2 < len(player) < 50:
                    if player not in companies: companies.append(player)
        
        seen = set(); unique_players = []
        companies.sort(key=len, reverse=True)
        for player in companies:
            is_substring = any(player in sp for sp in seen)
            if not is_substring: unique_players.append(player); seen.add(player)
        market_data["key_players"] = unique_players[:15]

        try:
            sentences = custom_sent_tokenize(combined_text)
            trend_keywords = ["innovation", "advancement", "growth in", "adoption of", "increasing demand", "shift towards", "emergence of", "rising popularity", "expanding use", "key trend", "driving factor", "opportunity", "development"]
            trends_found = [s.strip() for s in sentences if any(kw in s.lower() for kw in trend_keywords) and "market" in s.lower() and 30 < len(s.strip()) < 300]
            filtered_trends = [t for t in trends_found if not any(cw in t.lower() for cw in ["challenge", "obstacle", "risk"]) and re.search(r"\b(is|are|will be|expected to|driving|leading to|enabling)\b", t.lower())]
            market_data["trends"] = list(dict.fromkeys(filtered_trends))[:7]

            challenge_keywords = ["challenge", "obstacle", "barrier", "issue", "concern", "risk", "limitation", "constraint", "difficulty", "threat", "hindrance", "restraint"]
            challenges_found = [s.strip() for s in sentences if any(kw in s.lower() for kw in challenge_keywords) and "market" in s.lower() and 30 < len(s.strip()) < 300]
            market_data["challenges"] = list(dict.fromkeys(challenges_found))[:5]
        except Exception as e:
            st.sidebar.warning(f"Market data text processing error: {e}")
            if not market_data.get("trends"): market_data["trends"] = ["Trend extraction issue."]
            if not market_data.get("challenges"): market_data["challenges"] = ["Challenge extraction issue."]
        return market_data

    def extract_general_summary_and_keywords(self, combined_text, query_text, num_summary_sentences=7, num_keywords=10):
        # This function also remains largely the same.
        summary_data = {"summary_sentences": [], "keywords": [], "full_text_snippet": combined_text[:3000] + "..." if len(combined_text) > 3000 else combined_text}
        if not combined_text: return summary_data
        try:
            sentences = custom_sent_tokenize(combined_text)
            if not sentences: return summary_data
            stop_words = CUSTOM_STOPWORDS
            query_words_tokenized = custom_word_tokenize(query_text)
            query_content_words = [w for w in query_words_tokenized if w.isalnum() and w not in stop_words and len(w) > 2]
            sentence_scores = []
            for i, s_original in enumerate(sentences):
                s = s_original.strip()
                if not (30 < len(s) < 500): continue
                score = 0; sent_words = custom_word_tokenize(s)
                for qw in query_content_words:
                    if qw in sent_words: score += 2
                score += max(0, (10 - i) * 0.1)
                num_content_words = len([w for w in sent_words if w not in stop_words and len(w) > 3])
                score += num_content_words * 0.05
                if len(sent_words) > 0 and (len(sent_words) - num_content_words) / len(sent_words) > 0.7: score *= 0.5
                sentence_scores.append((s, score, i))
            sentence_scores.sort(key=lambda x: (-x[1], x[2]))
            selected_scored_sentences = [s_info[0] for s_info in sentence_scores]
            summary_data["summary_sentences"] = list(dict.fromkeys(selected_scored_sentences))[:num_summary_sentences]
            words = custom_word_tokenize(combined_text)
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 3]
            if filtered_words:
                word_counts = Counter(filtered_words)
                for qw in query_content_words:
                    if qw in word_counts: word_counts[qw] *= 1.5
                summary_data["keywords"] = [kw for kw, count in word_counts.most_common(num_keywords)]
        except Exception as e:
            st.sidebar.warning(f"General text processing error: {e}")
            if not summary_data.get("summary_sentences"): summary_data["summary_sentences"] = [combined_text[:500] + "..." if combined_text else ["Could not process text."]]
        return summary_data

    def simulate_search_results(self, query, num_results=3):
        # Unchanged, provides a fallback if DDGS fails or is not installed.
        query_lower = query.lower()
        results = []
        generic_results = [
            {'title': f'{query.capitalize()} General Overview (Simulated)', 'link': f'#simulated/{query_lower.replace(" ","-")}-overview', 
             'snippet': f'This is a simulated general overview of {query_lower}. It typically covers key aspects, definitions, and related information. For instance, if {query_lower} is a market, it might touch upon its size (e.g., USD 100 Billion in 2023) and growth rate (e.g., CAGR 10%). Key entities often mentioned include AlphaOrg and BetaCorp.'},
            {'title': f'Detailed Insights into {query.capitalize()} (Simulated)', 'link': f'#simulated/{query_lower.replace(" ","-")}-details', 
             'snippet': f'Further simulated details on {query_lower}, including discussions on its impact, relevance, and potential trends. Challenges could involve market saturation or technological disruption. Opportunities might arise from innovation.'},
            {'title': f'{query.capitalize()} Key Considerations & Players (Simulated)', 'link': f'#simulated/{query_lower.replace(" ","-")}-points', 
             'snippet': f'Important simulated points regarding {query_lower}. Some notable entities simulated are Gamma Inc., Delta LLC. These players might have significant market share.'}
        ]
        if "electric vehicle" in query_lower: results = [{'title': 'Simulated EV Market Overview', 'link': '#simulated/ev-market', 'snippet': 'Electric vehicles (EVs) are gaining traction globally. Simulated key players: Tesla, BYD. Simulated market size: USD 500 Billion (2023), with a CAGR of around 18%. Trends include battery tech advancements and charging infrastructure expansion.'}]
        elif "ai market" in query_lower or "artificial intelligence" in query_lower: results = [{'title': 'Simulated AI Market Summary', 'link': '#simulated/ai-market', 'snippet': 'Artificial intelligence (AI) is a rapidly evolving field. Simulated leaders: Google, Microsoft. Simulated CAGR: ~35%. Trends point to generative AI and cross-industry adoption.'}]
        else: results = generic_results
        return results[:num_results]

class QueryTopicResearcher: 
    # Largely unchanged, orchestrates web research and data merging.
    def __init__(self):
        self.web_researcher = WebResearcher()
        self.known_industries_keywords_map = { 
            "electric vehicle": ["electric vehicle", "ev market", "electric car", "bev", "phev", "electric mobility"],
            "artificial intelligence": ["artificial intelligence", "ai market", "machine learning", "deep learning", "generative ai", "ai industry"],
            "renewable energy": ["renewable energy", "solar power", "wind energy", "green energy", "clean energy market"],
            "cloud computing": ["cloud computing", "iaas", "paas", "saas", "cloud services market", "aws", "azure", "gcp"],
            "pharmaceutical": ["pharmaceutical market", "pharma industry", "drug development", "biotechnology market"],
            "fintech": ["fintech market", "financial technology", "digital payments", "blockchain finance"],
        }
        self.predefined_industry_data = {
             "electric vehicle": {
                "market_size": "USD 500 Billion (2023 Est.)", "cagr": "18.0%",
                "key_players": ["Tesla", "BYD", "Volkswagen", "SAIC", "Stellantis", "Mercedes-Benz", "BMW", "Ford", "General Motors", "Hyundai-Kia"],
                "trends": ["Rapid adoption of EVs globally.", "Advancements in battery technology and range.", "Expansion of charging infrastructure.", "Government incentives and regulations supporting EVs.", "Development of autonomous driving features in EVs."],
                "challenges": ["High initial cost of EVs compared to ICE vehicles.", "Range anxiety and charging time concerns.", "Supply chain constraints for batteries and semiconductors.", "Need for widespread and standardized charging infrastructure.", "Competition from traditional automakers transitioning to EVs."],
                "market_share": {"Tesla": 20, "BYD": 18, "Volkswagen Group": 12, "Stellantis": 7, "Hyundai-Kia": 7, "Others": 36},
                "regions": {"Asia-Pacific (esp. China)": 50, "Europe": 30, "North America": 15, "Rest of World": 5},
                "segments": {"Battery Electric Vehicles (BEV)": 75, "Plug-in Hybrid Electric Vehicles (PHEV)": 25},
                "forecast": { "Tesla": [20,19,18], "BYD": [18,19,20], "Volkswagen Group": [12,13,14], "Others": [50,49,48]},
                "opportunities": ["Growth in emerging markets.", "Development of more affordable EV models.", "Integration of vehicle-to-grid (V2G) technology.", "Battery recycling and second-life applications.", "Innovations in solid-state batteries."]
            },
            "artificial intelligence": {
                "market_size": "USD 200 Billion (2023 Est.)", "cagr": "37.0%",
                "key_players": ["Google (Alphabet)", "Microsoft", "Amazon (AWS)", "NVIDIA", "IBM", "Meta", "OpenAI", "Baidu", "Salesforce", "Intel"],
                "trends": ["Increasing adoption of AI/ML across various industries.", "Rapid growth of generative AI models and applications (e.g., LLMs).", "Focus on AI ethics, responsible AI, and explainability (XAI).", "Edge AI development for on-device processing.", "AI-powered automation in business processes."],
                "challenges": ["Data privacy, security, and bias concerns in AI models.", "Shortage of skilled AI talent and expertise.", "High cost of AI research, development, and infrastructure.", "Regulatory uncertainty and lack of standardized AI governance.", "Integrating AI seamlessly into existing workflows and systems."],
                "market_share": {"Google": 22, "Microsoft": 20, "Amazon (AWS)": 15, "NVIDIA": 10, "Others": 33},
                "regions": {"North America": 45, "Asia-Pacific": 30, "Europe": 20, "Rest of World": 5},
                "segments": {"AI Software (Platforms & Applications)": 60, "AI Hardware (Processors, GPUs)": 25, "AI Services (Consulting, Integration)": 15},
                "forecast": {"Google": [22,23,24], "Microsoft": [20,21,22], "NVIDIA": [10,12,14],"Others": [48,44,40]},
                "opportunities": ["Transforming industries like healthcare, finance, and manufacturing.", "Development of personalized customer experiences and services.", "Advancing scientific research and discovery (e.g., drug discovery).", "Improving efficiency and productivity through automation.", "New business models based on AI-as-a-Service."]
            },
        }

    def identify_query_topic_and_type(self, query_text):
        query_lower = query_text.lower()
        for industry_key, keywords_list in self.known_industries_keywords_map.items():
            for keyword in keywords_list:
                if keyword in query_lower:
                    if any(term in query_lower for term in ["report", "analysis", "trends", "market", "industry", "competitors", "outlook"]):
                        return industry_key, "industry_analysis"
        
        general_industry_terms = ["market", "industry", "sector", "cagr", "market share", "competitors", "trends analysis", "strategic report"]
        if any(term in query_lower for term in general_industry_terms):
            match = re.search(r"(?:for|of|on|about|analyze|investigate|report on)\s+(?:the\s+)?([\w\s\-]+?)\s+(?:market|industry|sector)", query_lower)
            if match:
                extracted_topic = match.group(1).strip()
                for industry_key_refined, keywords_list_refined in self.known_industries_keywords_map.items():
                    if extracted_topic == industry_key_refined or extracted_topic in keywords_list_refined:
                         return industry_key_refined, "industry_analysis"
                return extracted_topic, "industry_analysis"

        topic = re.sub(r"^(?:what is|tell me about|generate a report on|analyze|information on|provide details on|summarize)\s*(?:the\s+)?", "", query_lower, flags=re.IGNORECASE).strip()
        topic = topic.replace("?", "").replace(" report", "").replace(" analysis", "")
        return topic if topic else "general information", "general_information"

    def get_query_data(self, query_text_input):
        identified_topic, query_type = self.identify_query_topic_and_type(query_text_input)
        st.write(f"--- Identified query type: **{query_type.replace('_',' ').title()}** for topic: **'{identified_topic.title()}'** ---")

        is_industry = (query_type == "industry_analysis")
        web_specific_market_data, fetched_sources, combined_web_text = self.web_researcher.research_topic_online(identified_topic, is_industry)

        final_data_payload = {}

        if is_industry:
            final_data_payload = self.predefined_industry_data.get(identified_topic, {}).copy() 
            if not final_data_payload: 
                st.info(f"'{identified_topic.title()}' is analyzed as an industry. No specific pre-configuration found. Report relies on web search and generic templates. Quality may vary.")
                final_data_payload = {
                    "market_size": "", "cagr": "", "key_players": [], "trends": [], "challenges": [],
                    "market_share": {}, "regions": {}, "segments": {}, "forecast": {}, "opportunities": [],
                    "year": datetime.now().year, "base_year_market_size": "", "forecast_year_market_size": ""
                }
            
            if web_specific_market_data.get("market_size"): final_data_payload["market_size"] = web_specific_market_data["market_size"]
            if web_specific_market_data.get("base_year_market_size"): final_data_payload["base_year_market_size"] = web_specific_market_data["base_year_market_size"]
            if web_specific_market_data.get("forecast_year_market_size"): final_data_payload["forecast_year_market_size"] = web_specific_market_data["forecast_year_market_size"]
            if web_specific_market_data.get("cagr"): final_data_payload["cagr"] = web_specific_market_data["cagr"]
            
            for field in ["key_players", "trends", "challenges"]:
                web_values = web_specific_market_data.get(field, [])
                predefined_values = final_data_payload.get(field, [])
                combined_values = web_values + [pv for pv in predefined_values if pv not in web_values]
                final_data_payload[field] = list(dict.fromkeys(combined_values))[:10 if field == "key_players" else (7 if field == "trends" else 5)]

            final_data_payload.setdefault("key_players", ["Undetermined Key Players from Web Search"])
            final_data_payload.setdefault("trends", ["General industry developments observed."])
            final_data_payload.setdefault("challenges", ["Standard competitive pressures and operational hurdles."])
            final_data_payload.setdefault("opportunities", ["Exploration of new market segments.", "Technological integration."])
            final_data_payload.setdefault("market_share", {"Top Player (Est.)": 30, "Challenger (Est.)": 20, "Others": 50})
            final_data_payload.setdefault("regions", {"Primary Region (Global/Dominant)": 60, "Secondary": 30, "Others": 10})
            final_data_payload.setdefault("segments", {"Main Segment": 70, "Other Segments": 30})
            final_data_payload.setdefault("forecast", {"Overall Market Trend (Illustrative)": [100, 110, 121]})

        elif query_type == "general_information":
            general_summary_data = self.web_researcher.extract_general_summary_and_keywords(combined_web_text, query_text_input)
            final_data_payload = general_summary_data
            final_data_payload['raw_text_sample'] = combined_web_text[:2000]

        final_data_payload["fetched_sources"] = fetched_sources
        final_data_payload["query_type"] = query_type
        final_data_payload["query_topic"] = identified_topic 
        final_data_payload["original_query"] = query_text_input
        final_data_payload["raw_web_text_for_analysis"] = combined_web_text
        
        return final_data_payload

# DataAnalyzer, ReportGenerator, ReportVisualizer classes are unchanged from the previous version.
# They operate on the data provided by QueryTopicResearcher.
class DataAnalyzer: 
    def analyze_market_trends(self, industry_data):
        market_size = industry_data.get("market_size") or industry_data.get("forecast_year_market_size") or industry_data.get("base_year_market_size") or "N/A"
        growth_rate = industry_data.get("cagr", "N/A")
        trends = industry_data.get("trends", [])
        
        analysis = {
            "market_size": market_size, "growth_rate": growth_rate,
            "key_trends": trends if trends else ["No specific trends identified from available data."],
            "trend_strength": [random.randint(60, 95) for _ in range(len(trends if trends else [""]))], # Simulated strength
            "summary": f"The market, valued around {market_size}, is reportedly growing at a CAGR of approximately {growth_rate}."
        }
        return analysis

    def analyze_competitors(self, industry_data):
        key_players = industry_data.get("key_players", ["Player A (Generic)", "Player B (Generic)"])
        market_share_data = industry_data.get("market_share", {"Generic Player A": 60, "Others": 40})
        
        valid_shares = {k: v for k, v in market_share_data.items() if k.lower() != "others" and isinstance(v, (int, float)) and v > 0}
        sorted_shares = sorted(valid_shares.items(), key=lambda x: x[1], reverse=True)
        
        cr4 = sum(s[1] for s in sorted_shares[:4]) if sorted_shares else 0
        
        structure = "Competitive"
        if cr4 == 0 and len(key_players) > 5 : structure = "Fragmented (CR4 unknown)"
        elif cr4 > 80: structure = "Highly Concentrated (Oligopoly)"
        elif cr4 > 60: structure = "Concentrated"
        elif cr4 > 40: structure = "Moderately Concentrated"
        elif cr4 > 0 : structure = "Moderately Competitive"
            
        analysis = {
            "key_players": key_players if key_players else ["No specific players identified."],
            "market_share_data": market_share_data, 
            "market_concentration": {"cr4": cr4, "structure": structure},
            "leader_advantage_points": (sorted_shares[0][1] - sorted_shares[1][1]) if len(sorted_shares) > 1 else (sorted_shares[0][1] if sorted_shares else 0),
            "top_players_list": [name for name, _ in sorted_shares[:5]] if sorted_shares else (key_players[:3] if key_players and key_players[0] != "Undetermined Key Players from Web Search" else ["N/A"])
        }
        return analysis
        
    def analyze_regional_impact(self, industry_data):
        regions_data = industry_data.get("regions", {"Global Focus (Undetermined Detail)": 100})
        dominant_region_tuple = max(regions_data.items(), key=lambda x: x[1]) if regions_data and any(isinstance(v, (int,float)) and v > 0 for v in regions_data.values()) else ("N/A", 0) # check for actual values
        dominant_region = dominant_region_tuple[0]
        
        cagr_str = str(industry_data.get("cagr","5%")).replace('%','').replace('N/A','5')
        try: cagr_val = float(cagr_str)
        except ValueError: cagr_val = 5.0

        regional_growth_rates = {r: round(random.uniform(max(1, cagr_val * 0.7), cagr_val * 1.3), 1) for r in regions_data.keys()}
        fastest_growing_region_tuple = max(regional_growth_rates.items(), key=lambda x: x[1]) if regional_growth_rates else ("N/A", 0)
        fastest_growing_region = fastest_growing_region_tuple[0]
        
        emerging_markets_list = [r for r,s_val in regions_data.items() if isinstance(s_val, (int,float)) and s_val < 20 and regional_growth_rates.get(r,0) > (cagr_val * 0.9)]

        analysis = {
            "regional_distribution_data": regions_data, 
            "dominant_region_name": dominant_region,
            "dominant_region_share": dominant_region_tuple[1],
            "regional_growth_rates": regional_growth_rates, 
            "fastest_growing_region_name": fastest_growing_region,
            "fastest_growing_region_rate": fastest_growing_region_tuple[1],
            "potential_emerging_markets": emerging_markets_list if emerging_markets_list else ["Further analysis needed for emerging markets."]
        }
        return analysis
        
    def analyze_segments(self, industry_data):
        segments_data = industry_data.get("segments", {"Primary Segment (Generic)": 100})
        dominant_segment_tuple = max(segments_data.items(), key=lambda x: x[1]) if segments_data and any(isinstance(v, (int,float)) and v > 0 for v in segments_data.values()) else ("N/A", 0)
        dominant_segment = dominant_segment_tuple[0]

        cagr_str = str(industry_data.get("cagr","5%")).replace('%','').replace('N/A','5')
        try: cagr_val = float(cagr_str)
        except ValueError: cagr_val = 5.0

        segment_growth_rates = {s: round(random.uniform(max(1, cagr_val * 0.8), cagr_val * 1.5), 1) for s in segments_data.keys()}
        fastest_growing_segment_tuple = max(segment_growth_rates.items(), key=lambda x: x[1]) if segment_growth_rates else ("N/A", 0)
        fastest_growing_segment = fastest_growing_segment_tuple[0]
        
        analysis = {
            "segment_distribution_data": segments_data, 
            "dominant_segment_name": dominant_segment,
            "dominant_segment_share": dominant_segment_tuple[1],
            "segment_growth_rates": segment_growth_rates, 
            "fastest_growing_segment_name": fastest_growing_segment,
            "fastest_growing_segment_rate": fastest_growing_segment_tuple[1]
        }
        return analysis
        
    def analyze_future_outlook(self, industry_data):
        forecast_data = industry_data.get("forecast", {"Overall Market Index": [100,110,121,133]})
        if not isinstance(forecast_data, dict) or not forecast_data: forecast_data = {"Overall Market Index": [100,110,121,133]}
        
        for k,v_list in forecast_data.items(): 
            if not isinstance(v_list, list) or not all(isinstance(x,(int,float)) for x in v_list) or len(v_list) < 2:
                start_val = random.randint(50,150)
                forecast_data[k] = [round(start_val * (1 + random.uniform(-0.05, 0.15))**i) for i in range(random.randint(3,5))]

        challenges_list = industry_data.get("challenges", ["Generic challenge: Economic uncertainty.", "Generic challenge: Adapting to new technologies."])
        opportunities_list = industry_data.get("opportunities", ["Generic opportunity: Market expansion.", "Generic opportunity: Leveraging data."])
        
        trend_analysis_map = {}
        for co_name, data_series in forecast_data.items():
            if len(data_series)>1:
                abs_trend = data_series[-1] - data_series[0]
                percent_change = round((abs_trend) / data_series[0] * 100, 1) if data_series[0] != 0 else 0
                direction = "increasing" if abs_trend > 0 else ("decreasing" if abs_trend < 0 else "stable")
                trend_analysis_map[co_name] = {"data_series": data_series, "absolute_change": abs_trend, "direction": direction, "percentage_change": percent_change}
        
        growing_entities = sorted([(k,v["percentage_change"]) for k,v in trend_analysis_map.items() if v["percentage_change"] > 0 and k.lower() != "others" and "overall" not in k.lower()], key=lambda x:x[1], reverse=True)
        
        analysis = {
            "forecast_chart_data": forecast_data, "forecast_trend_analysis": trend_analysis_map,
            "fastest_growing_forecasted_entities": [name for name,_ in growing_entities[:3]] if growing_entities else ["N/A - Detailed player forecasts limited."],
            "key_challenges_outlook": challenges_list[:3], "key_opportunities_outlook": opportunities_list[:3]
        }
        return analysis

class ReportGenerator:
    # All generate_... methods are unchanged from the previous comprehensive version.
    # These methods format the analyzed data into Markdown.
    def generate_executive_summary(self, topic_name, market_analysis, competitor_analysis, future_analysis):
        market_size = market_analysis.get("market_size", "N/A")
        growth_rate = market_analysis.get("growth_rate", "N/A")
        top_players_list = competitor_analysis.get("top_players_list", ["Key industry participants"])
        top_players_str = ", ".join(top_players_list[:3]) if top_players_list and top_players_list[0] != "N/A" else "leading industry participants"
        market_structure = competitor_analysis.get("market_concentration", {}).get("structure", "competitive")
        cr4_value = competitor_analysis.get("market_concentration", {}).get("cr4", 0)
        cr4_text = f"with top four players accounting for ~{cr4_value:.1f}% market share" if cr4_value > 0 else "indicating a potentially fragmented or data-limited landscape"
        key_trends_list = market_analysis.get("key_trends", ["general industry developments"])
        trend1 = key_trends_list[0].lower().strip('.') if key_trends_list else "general industry developments"
        key_opportunities_list = future_analysis.get("key_opportunities_outlook", ["strategic growth areas"])
        opportunity1 = key_opportunities_list[0].lower().strip('.') if key_opportunities_list else "strategic growth areas"

        return f"""## Executive Summary
The **{topic_name.title()}** industry presents a dynamic landscape. Valued at ~**{market_size}**, it is projected to grow at a CAGR of ~**{growth_rate}**. 
Key insights:
1.  **Market Structure**: A **{market_structure}** market, {cr4_text}.
2.  **Leading Companies**: **{top_players_str}** are notable.
3.  **Growth Drivers**: **{trend1}** is a significant factor.
4.  **Opportunities**: Potential in areas like **{opportunity1}**.
This report offers strategic insights for navigating the {topic_name.title()} market."""

    def generate_market_overview(self, topic_name, ma, ra, sa): # Using abbreviations for clarity
        market_size_text = ma.get("market_size","N/A"); growth_rate_text = ma.get("growth_rate","N/A")
        dr_name = ra.get("dominant_region_name","N/A"); dr_share = ra.get("dominant_region_share","N/A")
        fgr_name = ra.get("fastest_growing_region_name","N/A"); fgr_rate = ra.get("fastest_growing_region_rate","N/A")
        regional_text = f"{dr_name} leads (~{dr_share}% share). Fastest growing: {fgr_name} (~{fgr_rate}% CAGR)." if dr_name != "N/A" else "Regional data limited."
        
        ds_name = sa.get("dominant_segment_name","N/A"); ds_share = sa.get("dominant_segment_share","N/A")
        fgs_name = sa.get("fastest_growing_segment_name","N/A"); fgs_rate = sa.get("fastest_growing_segment_rate","N/A")
        segment_text = f"Dominant segment: {ds_name} (~{ds_share}%). Fastest growing: {fgs_name} (~{fgs_rate}% CAGR)." if ds_name != "N/A" else "Segment data limited."
        segments_list = ", ".join(list(sa.get("segment_distribution_data",{}).keys())) if sa.get("segment_distribution_data",{}) else 'various categories'

        drivers = ma.get("key_trends", [])[:3]; drivers_md = "\n".join([f"    - {d.strip('.')}" for d in drivers]) if drivers else "    - General economic and technological factors."

        return f"""## Market Overview
### Market Size & Growth
**{topic_name.title()}** market: ~**{market_size_text}**, CAGR: ~**{growth_rate_text}**.
### Regional Insights
{regional_text}
### Key Segments
{segment_text} Segments include: {segments_list}.
### Primary Drivers
{drivers_md}"""

    def generate_competitor_analysis(self, topic_name, ca): # Using ca abbreviation
        mc = ca.get("market_concentration",{}); market_structure = mc.get("structure","competitive")
        cr4 = mc.get("cr4",0); cr4_desc = f"Top 4 firms: ~{cr4:.1f}% share." if cr4 > 0 else "Concentration data limited."
        
        players = ca.get("key_players", [])[:8]; players_md = "\n".join([f"- **{p.strip()}**" for p in players]) if players and players != ["Undetermined Key Players from Web Search"] else "- Specific player data limited."
        
        leader_adv = ca.get("leader_advantage_points",0)
        dynamics = f"Dynamics are {'intense' if 0 < leader_adv < 10 else 'variable'}. Leader advantage: ~{leader_adv:.1f} pts." if leader_adv > 0 or cr4 > 0 else "Competitive dynamics analysis requires more granular data."

        return f"""## Competitor Analysis
### Market Concentration
**{topic_name.title()}** market: **{market_structure}**. {cr4_desc}
### Key Players
{players_md}
*(Note: List based on available data, may not be exhaustive.)*
### Competitive Dynamics
{dynamics} Competition on innovation, price, brand."""

    def generate_trends_analysis(self, topic_name, ma): # Using ma abbreviation
        trends = ma.get("key_trends", [])[:5]; strengths = ma.get("trend_strength", [])
        intro = f"## Key Industry Trends & Developments\nKey trends shaping the **{topic_name.title()}** market:"
        if not trends or trends == ["No specific trends identified from available data."]:
            return intro + "\n\n- Detailed trend analysis limited. General evolution includes tech, consumer shifts, sustainability, regulations."
        trends_md = ""
        for i, trend in enumerate(trends):
            s_val = strengths[i] if i < len(strengths) else random.randint(70,90)
            imp = "High" if s_val > 85 else ("Medium" if s_val > 70 else "Moderate")
            trends_md += f"\n### {i+1}. {trend.strip('.')}\n   - **Significance**: {imp} impact (Est. {s_val}/100). Implication: Adapt by [e.g., tech investment, strategy revision].\n"
        return intro + trends_md

    def generate_strategic_recommendations(self, topic_name, ma, ca, fa): # Using abbreviations
        ops = fa.get("key_opportunities_outlook", ["tech integration", "underserved segments"])[:2]
        chs = fa.get("key_challenges_outlook",["economic volatility", "regulatory scrutiny"])[:2]
        cr4 = ca.get("market_concentration",{}).get("cr4",50); struct = ca.get("market_concentration",{}).get("structure","competitive")
        recs = []
        if "concentrated" in struct.lower() or cr4 > 60: recs.append(f"**Positioning**: In a {struct.lower()} market, focus on differentiation, niche penetration, or alliances.")
        elif "fragmented" in struct.lower() or (0 < cr4 < 40): recs.append(f"**Positioning**: In a {struct.lower()} landscape, pursue share growth via innovation, customer acquisition, M&A.")
        else: recs.append(f"**Positioning**: Assess landscape; build brand equity, customer loyalty. Adaptability is key.")
        recs.append(f"**Leverage Opportunities**: Prioritize **{ops[0].lower().strip('.')}**. Develop roadmaps.")
        recs.append(f"**Address Challenges**: Mitigate risks from **{chs[0].lower().strip('.')}** (e.g., diversify supply chains, enhance resilience).")
        tech_driven = any(any(t_kw in trend.lower() for t_kw in ["tech", "digital", "ai", "auto", "inno"]) for trend in ma.get("key_trends",[]))
        recs.append(f"**Innovation**: {'Accelerate R&D and tech adoption.' if tech_driven else 'Drive continuous improvement and operational excellence.'}")
        recs.append(f"**Customer Centricity**: Deepen understanding of needs in {topic_name.title()} market. Invest in CX.")
        recs.append(f"**Alliances**: Explore partnerships for new markets/tech, risk sharing, esp. for {chs[1].lower().strip('.') if len(chs)>1 else 'market entry barriers'}.")
        recs_md = "\n".join([f"{i+1}. {r}" for i, r in enumerate(recs)])
        return f"""## Strategic Recommendations
For **{topic_name.title()}** market:
{recs_md}
*Disclaimer: High-level, AI-derived recommendations. Validate with primary research.*"""

    def generate_future_outlook(self, topic_name, fa): # Using fa abbreviation
        fgp = fa.get("fastest_growing_forecasted_entities",[])
        dyn = f"Market may see competition from agile players like {', '.join(fgp)}, challenging leaders." if fgp and not fgp[0].startswith("N/A") else "Market likely to follow current trajectory; innovators may disrupt."
        kco = fa.get("key_challenges_outlook", []); kco_md = "Challenges: " + "\n".join([f"    - {c.strip('.')}" for c in kco]) if kco else "Ongoing adaptation to economic/tech shifts."
        koo = fa.get("key_opportunities_outlook", []); koo_md = "Opportunities: " + "\n".join([f"    - {o.strip('.')}" for o in koo]) if koo else "Exploiting tech and evolving consumer needs."
        unc1 = kco[0].lower().strip('.') if kco else "pace of tech change"
        unc2 = kco[1].lower().strip('.') if len(kco) > 1 else "global economic/regulatory conditions"
        return f"""## Future Market Outlook
### Market Evolution (3-5 Years)
**{topic_name.title()}** market: {dyn} Sustainability, digitalization, consumer expectations will be key.
### Opportunities on Horizon
{koo_md}
### Persistent/Emerging Challenges
{kco_md}
### Critical Uncertainties
1.  **Impact of {unc1.capitalize()}**.
2.  **Influence of {unc2.capitalize()}**.
Proactive monitoring and scenario planning advised."""

    def generate_general_introduction(self, original_query, topic_name):
        return f"""## Introduction
Report on: "{original_query}". Focus: **{topic_name.title()}**, based on public web content.
"""

    def generate_general_summary(self, summary_sentences):
        if not summary_sentences or not isinstance(summary_sentences, list) or not summary_sentences[0]:
            return "## Key Information Summary\nNo specific summary points reliably extracted.\n"
        points = "\n".join([f"- {s.strip()}" for s in summary_sentences])
        return f"""## Key Information Summary
Extracted for **{st.session_state.current_topic_name.title()}**:
{points}"""

    def generate_general_keywords(self, keywords):
        if not keywords or not isinstance(keywords, list) or not keywords[0]:
            return "\n## Main Keywords/Topics Identified\nNo distinct keywords prominently identified.\n"
        kw_md = "\n".join([f"- {kw.capitalize()}" for kw in keywords])
        return f"""## Main Keywords/Topics Identified
For **{st.session_state.current_topic_name.title()}**:
{kw_md}"""
    
    def generate_text_snippet_section(self, text_snippet):
        if not text_snippet or len(text_snippet) < 100: return ""
        return f"""\n## Extended Content Snippet
Aggregated snippet:
\n---\n{text_snippet}\n---"""

    def compile_full_report(self, topic_name, sections, fetched_sources=None, query_type="industry_analysis"):
        date_str = datetime.now().strftime("%B %d, %Y")
        title_main = f"{topic_name.title()} Information Report"; subtitle = "Key Insights from Automated Web Research"
        toc_map = {"Introduction": "introduction", "Key Information Summary": "key-information-summary", "Main Keywords/Topics Identified": "main-keywords-topics-identified"}

        if query_type == "industry_analysis":
            title_main = f"{topic_name.title()} Industry Intelligence Report"; subtitle = "Comprehensive Market Analysis & Strategic Recommendations (AI-Generated)"
            toc_map = {"Executive Summary": "executive-summary", "Market Overview": "market-overview", "Competitor Analysis": "competitor-analysis", "Key Industry Trends & Developments": "key-industry-trends-developments", "Strategic Recommendations": "strategic-recommendations", "Future Market Outlook": "future-market-outlook"}

        title_md = f"# {title_main}\n### {subtitle}\n*Generated: {date_str}*\n"
        toc_md = "\n## Table of Contents\n" + "\n".join([f"{i+1}. [{name}](#{anchor})" for i, (name, anchor) in enumerate(toc_map.items())])
        
        appendix_num = len(toc_map) + 1
        if fetched_sources: toc_md += f"\n{appendix_num}. [Data Sources Appendix](#data-sources-appendix)\n"
        full_report = title_md + toc_md + "".join(sections)
        
        if fetched_sources:
            sources_md = f"\n## Data Sources Appendix\nSynthesized from (accessed {date_str}):\n"
            sources_md += "\n".join([f"{i+1}. **[{re.sub(r'[\[\]]', '', s.get('title','Source'))}]({s.get('link','#')})**" for i, s in enumerate(fetched_sources)])
            full_report += sources_md
        
        disclaimer = """\n\n---\n**Disclaimer:** *AI-generated report from public web data & predefined datasets. Accuracy not guaranteed. For informational/preliminary research only, not sole basis for decisions. Verify independently. Content reflects data at generation time.*"""
        full_report += disclaimer
        return full_report

class ReportVisualizer:
    # All create_..._chart methods are unchanged from the previous comprehensive version.
    # They generate matplotlib charts based on the analyzed data.
    def create_market_share_chart(self, market_share_data, topic_name=""):
        title = f"{topic_name.title()} Market Share (%)" if topic_name else "Market Share (%)"
        if not market_share_data or not isinstance(market_share_data, dict) or sum(v for v in market_share_data.values() if isinstance(v, (int,float))) == 0:
            return generate_placeholder_image(text=f"Market Share Data N/A for {topic_name}")
        valid_shares = {k: v for k,v in market_share_data.items() if isinstance(v,(int,float)) and v > 0.1}
        if not valid_shares: return generate_placeholder_image(text=f"No Valid Market Share Data for {topic_name}")
        sorted_items = sorted(valid_shares.items(), key=lambda x: x[1], reverse=True)
        max_slices = 6
        chart_data_list = sorted_items[:max_slices-1] + [("Others", sum(s for _,s in sorted_items[max_slices-1:]))] if len(sorted_items) > max_slices else sorted_items
        labels = [i[0] for i in chart_data_list]; values = [i[1] for i in chart_data_list]
        return create_pie_chart(title, labels, values)
        
    def create_regional_distribution_chart(self, regions_data, topic_name=""):
        title = f"{topic_name.title()} Regional Distribution (%)" if topic_name else "Regional Distribution (%)"
        if not regions_data or not isinstance(regions_data, dict) or sum(v for v in regions_data.values() if isinstance(v, (int,float))) == 0:
            return generate_placeholder_image(text=f"Regional Data N/A for {topic_name}")
        return create_bar_chart(title, "Region", "Market Share (%)", list(regions_data.keys()), list(regions_data.values()))
        
    def create_segment_distribution_chart(self, segments_data, topic_name=""):
        title = f"{topic_name.title()} Segment Distribution (%)" if topic_name else "Segment Distribution (%)"
        if not segments_data or not isinstance(segments_data, dict) or sum(v for v in segments_data.values() if isinstance(v, (int,float))) == 0:
            return generate_placeholder_image(text=f"Segment Data N/A for {topic_name}")
        return create_pie_chart(title, list(segments_data.keys()), list(segments_data.values()))
        
    def create_forecast_chart(self, forecast_data, topic_name=""):
        title = f"{topic_name.title()} Forecast (Illustrative)" if topic_name else "Forecast (Illustrative)"
        if not forecast_data or not isinstance(forecast_data, dict) or not any(isinstance(v,list) and len(v)>1 for v in forecast_data.values()):
            return generate_placeholder_image(text=f"Forecast Data N/A for {topic_name}")
        valid_fc_data = {k:v for k,v in forecast_data.items() if isinstance(v, list) and all(isinstance(x,(int,float)) for x in v) and len(v)>1}
        if not valid_fc_data: return generate_placeholder_image(text=f"No Valid Forecast Series for {topic_name}")
        plot_data = {}
        if "Overall Market Index" in valid_fc_data: plot_data["Overall Market Index"] = valid_fc_data["Overall Market Index"]
        other_series = {k:v for k,v in valid_fc_data.items() if k != "Overall Market Index"}
        sorted_others = sorted(other_series.items(), key=lambda item: item[1][-1], reverse=True)
        for i in range(min(len(sorted_others), 4 - len(plot_data))): plot_data[sorted_others[i][0]] = sorted_others[i][1]
        if not plot_data: plot_data = dict(list(valid_fc_data.items())[:4])
        return create_line_chart(title, "Time Period", "Value/Index", plot_data)

    def create_keywords_list_image(self, keywords, topic_name="", width=400, height=350):
        fig, ax = plt.subplots(figsize=(width/100, height/100)); title = f"Key Topics for {topic_name.title()}" if topic_name else "Key Topics/Keywords"
        if not keywords or not isinstance(keywords, list) or not keywords[0]:
            ax.text(0.5, 0.5, "No Keywords Extracted", ha='center', va='center', fontsize=12)
        else:
            ax.set_title(title, fontsize=14, pad=15); y_pos = 0.90; x1, x2 = 0.05, 0.55
            per_col = (len(keywords[:12]) + 1) // 2
            for i, kw in enumerate(keywords[:12]):
                curr_x = x1 if i < per_col else x2
                curr_y = y_pos - (i % per_col) * 0.09
                ax.text(curr_x, curr_y, f"- {kw.capitalize()}", fontsize=10, va='top')
        ax.axis('off'); plt.tight_layout(pad=1.5); buf = io.BytesIO()
        plt.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
        return buf

class InformationSynthesisSystem: 
    def __init__(self):
        self.researcher = QueryTopicResearcher()
        self.analyzer = DataAnalyzer() 
        self.generator = ReportGenerator()
        self.visualizer = ReportVisualizer()
        
    def process_query(self, query_text):
        st.session_state.current_step_progress_text = "Interpreting query & fetching data..."
        processed_data = self.researcher.get_query_data(query_text)
        
        query_type = processed_data.get("query_type", "general_information")
        report_topic = processed_data.get("query_topic", "General Topic").strip()
        if not report_topic: report_topic = "Undetermined Topic"
        st.session_state.current_topic_name = report_topic

        sections, viz, summary_ui = [], {}, {}

        if query_type == "industry_analysis":
            st.session_state.current_step_progress_text = "Analyzing market data..."
            ma_res = self.analyzer.analyze_market_trends(processed_data)
            ca_res = self.analyzer.analyze_competitors(processed_data)
            ra_res = self.analyzer.analyze_regional_impact(processed_data)
            sa_res = self.analyzer.analyze_segments(processed_data)
            fo_res = self.analyzer.analyze_future_outlook(processed_data)
            
            st.session_state.current_step_progress_text = "Generating report sections..."
            sections.extend([
                self.generator.generate_executive_summary(report_topic, ma_res, ca_res, fo_res),
                self.generator.generate_market_overview(report_topic, ma_res, ra_res, sa_res),
                self.generator.generate_competitor_analysis(report_topic, ca_res),
                self.generator.generate_trends_analysis(report_topic, ma_res),
                self.generator.generate_strategic_recommendations(report_topic, ma_res, ca_res, fo_res),
                self.generator.generate_future_outlook(report_topic, fo_res)
            ])
            
            st.session_state.current_step_progress_text = "Creating visualizations..."
            viz["market_share"] = self.visualizer.create_market_share_chart(ca_res.get("market_share_data"), report_topic)
            viz["regional"] = self.visualizer.create_regional_distribution_chart(ra_res.get("regional_distribution_data"), report_topic)
            viz["segment"] = self.visualizer.create_segment_distribution_chart(sa_res.get("segment_distribution_data"), report_topic)
            viz["forecast"] = self.visualizer.create_forecast_chart(fo_res.get("forecast_chart_data"), report_topic)

            summary_ui = {"Market Size": ma_res.get("market_size","N/A"), "CAGR": ma_res.get("growth_rate","N/A"), "Players": len(processed_data.get("key_players",[])), "Sources": len(processed_data.get("fetched_sources",[]))}
        
        elif query_type == "general_information":
            st.session_state.current_step_progress_text = "Extracting summary & keywords..."
            sections.extend([
                self.generator.generate_general_introduction(processed_data.get("original_query","N/A"), report_topic),
                self.generator.generate_general_summary(processed_data.get("summary_sentences")),
                self.generator.generate_general_keywords(processed_data.get("keywords"))
            ])
            st.session_state.current_step_progress_text = "Creating keyword visualization..."
            viz["keywords_list"] = self.visualizer.create_keywords_list_image(processed_data.get("keywords", []), report_topic)
            summary_ui = {"Summary Sentences": len(processed_data.get("summary_sentences",[])), "Keywords": len(processed_data.get("keywords",[])), "Sources": len(processed_data.get("fetched_sources",[]))}

        st.session_state.current_step_progress_text = "Compiling final report..."
        full_report_md = self.generator.compile_full_report(report_topic, sections, processed_data.get("fetched_sources"), query_type)
        
        return {"report_md": full_report_md, "topic_name": report_topic, "query_type": query_type, "visualizations": viz, "ui_summary_metrics": summary_ui}

# --- Documentation Strings ---
APPROACH_EXPLANATION_MD = """
### Approach Explanation
This AI system generates reports using a modular, multi-stage process:
1.  **Hybrid Data Strategy**: Combines predefined knowledge (e.g., for EV, AI industries) with real-time web scraping (DuckDuckGo Search or simulated).
2.  **Rule-Based NLP & Extraction**: Uses custom regex-based utilities for sentence/word tokenization and data extraction (market size, CAGR, etc.), avoiding heavy NLTK dependencies.
3.  **Graceful Degradation**: Falls back to simulated search or uses snippets if web fetching fails or libraries are missing.
4.  **Separation of Concerns**: Classes for Query Interpretation, Web Research, Data Analysis, Report Generation, and Visualization.
5.  **Progressive Report Building**: Augments predefined data with web-extracted info; relies on web data/templates for unknown topics.
6.  **User-Focused Output**: Markdown reports with visualizations and actionable recommendations (for industry reports).
The system aims for robustness and transparency, providing research starting points while disclaiming its automated nature.
"""
SYSTEM_DOCUMENTATION_MD = """
### System Architecture & Workflow
**Core Components:**
- **`InformationSynthesisSystem`**: Main orchestrator.
- **`QueryTopicResearcher`**: Interprets query, manages data acquisition (web + predefined), consolidates data.
- **`WebResearcher`**: Handles web search (DDGS/simulated), content fetching (`requests`), HTML parsing (`BeautifulSoup`), initial data extraction (regex for metrics, custom NLP for summaries/keywords).
- **`DataAnalyzer`**: (For industry reports) Performs deeper analysis on market trends, competitors, regions, segments, future outlook.
- **`ReportGenerator`**: Constructs report sections in Markdown.
- **`ReportVisualizer`**: Creates `matplotlib` charts.
- **Custom NLP Utilities**: `custom_sent_tokenize`, `custom_word_tokenize`, `CUSTOM_STOPWORDS`.

**Workflow:**
1.  User query input.
2.  `InformationSynthesisSystem` orchestrates:
    a.  `QueryTopicResearcher`: Identifies query type/topic.
    b.  `WebResearcher`: Fetches and processes web content. Data merged with predefined if applicable.
    c.  `DataAnalyzer` (if industry): Derives insights.
    d.  `ReportGenerator`: Creates textual sections.
    e.  `ReportVisualizer`: Generates charts.
    f.  `ReportGenerator.compile_full_report`: Assembles final Markdown.
3.  Streamlit UI displays report, visuals, and download options.
"""

# Streamlit UI (largely unchanged from previous, relies on updated classes)
def main():
    st.set_page_config(page_title="AI Research & Report Generator", layout="wide")
    
    st.sidebar.title("🧠 AI Report Generator")
    st.sidebar.markdown("Automated insights from web data.")
    
    if not DUCKDUCKGO_SEARCH_AVAILABLE:
        st.sidebar.warning("`duckduckgo_search` lib not found. Web search will be **simulated**. For live results: `pip install duckduckgo-search`", icon="⚠️")

    # Initialize session state variables
    for key, default_value in [('current_topic_name', "Topic"), 
                               ('query_input', ""), 
                               ('current_step_progress_text', "Initializing...")]:
        if key not in st.session_state:
            st.session_state[key] = default_value

    @st.cache_resource
    def load_system(): return InformationSynthesisSystem()
    system = load_system()
    
    st.title("AI Research & Report Generator")
    st.subheader("Input a business-level query for industry analysis or any topic for a general summary.")
    
    app_tab, doc_tab = st.tabs(["📄 Report Generation", "ℹ️ System Info & Documentation"])

    with app_tab:
        st.markdown("### Enter Your Query")
        query_examples = [
            "Comprehensive analysis of the electric vehicle market",
            "What are the key trends in the cloud computing industry?",
            "Report on the renewable energy sector: growth, players, and outlook",
            "AI market strategic recommendations",
            "Summarize the impact of climate change on agriculture"
        ]
        
        st.session_state.query_input = st.text_input(
            "E.g., 'cybersecurity market trends' or 'benefits of renewable energy'", 
            value=st.session_state.query_input, 
            key="query_text_field",
            placeholder="Type your research query here..."
        )
        
        st.markdown("<p style='font-size: smaller;'><b>Example Queries:</b></p>", unsafe_allow_html=True)
        num_example_cols = 3 
        example_cols = st.columns(num_example_cols)
        for i, example in enumerate(query_examples):
            if example_cols[i % num_example_cols].button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.query_input = example
                st.rerun()

        generate_button = st.button("🚀 Generate Report", type="primary", use_container_width=True, disabled=(not st.session_state.query_input.strip()))
        
        if generate_button and st.session_state.query_input.strip():
            query_to_process = st.session_state.query_input
            st.session_state.current_step_progress_text = "Initializing..."
            
            progress_bar_ui = st.progress(0, text=st.session_state.current_step_progress_text)
            start_time = time.time()
            
            # Update progress at key stages within process_query via st.session_state
            # The process_query method itself will update st.session_state.current_step_progress_text
            # We will try to update the progress bar based on that.
            # This is still a simplification as process_query is mostly monolithic.
            
            def update_spinner_and_progress(progress_value):
                progress_bar_ui.progress(progress_value, text=st.session_state.current_step_progress_text)

            update_spinner_and_progress(5) # Initializing
            
            # The system.process_query method now updates st.session_state.current_step_progress_text internally
            # We rely on those updates for the text part of the progress bar.
            # The numeric progress here is a rough guide.
            with st.spinner(f"Processing: {st.session_state.current_step_progress_text}"):
                result = system.process_query(query_to_process)
                update_spinner_and_progress(30) # After get_query_data
                # Simulate other stages if not explicitly broken down in process_query for progress
                if result['query_type'] == "industry_analysis":
                    update_spinner_and_progress(60) # After analysis
                else:
                    update_spinner_and_progress(70) # After general processing
                update_spinner_and_progress(85) # After report/viz generation
            
            progress_bar_ui.progress(100, text="Report finalized!")
            time.sleep(0.5); progress_bar_ui.empty()
            
            end_time = time.time()
            st.success(f"📊 Report for '{result['topic_name'].title()}' ({result['query_type'].replace('_',' ').title()}) generated in {end_time - start_time:.2f}s!", icon="✅")
            
            st.markdown("#### Quick Report Metrics:")
            sum_metrics = result["ui_summary_metrics"]
            metric_cols = st.columns(len(sum_metrics) if sum_metrics else 1)
            for i, (key, val) in enumerate(sum_metrics.items()):
                metric_cols[i % len(sum_metrics)].metric(key, str(val))

            report_display_tab, visuals_tab, export_tab = st.tabs(["📝 Full Report", "📊 Visualizations", "💾 Export Options"])
            
            with report_display_tab:
                st.markdown(result["report_md"], unsafe_allow_html=True)
                
            with visuals_tab:
                st.subheader(f"Visual Insights for {result['topic_name'].title()}")
                if result["query_type"] == "industry_analysis" and result["visualizations"]:
                    vis_col1, vis_col2 = st.columns(2)
                    charts_to_display = ["market_share", "segment", "regional", "forecast"]
                    cols = [vis_col1, vis_col2]
                    captions = {
                        "market_share": "Market Share Distribution", "segment": "Market Segments",
                        "regional": "Regional Distribution", "forecast": "Market Forecast (Illustrative)"
                    }
                    for i, chart_key in enumerate(charts_to_display):
                        if chart_key in result["visualizations"]:
                            with cols[i % 2]:
                                st.image(result["visualizations"][chart_key], caption=captions.get(chart_key, chart_key.replace("_", " ").title()), use_column_width=True)
                                st.markdown(get_image_download_link(result["visualizations"][chart_key], f"{result['topic_name'].replace(' ','_')}_{chart_key}.png"), unsafe_allow_html=True)
                elif result["query_type"] == "general_information" and "keywords_list" in result["visualizations"]:
                    st.image(result["visualizations"]["keywords_list"], caption="Extracted Keywords/Topics", use_column_width='auto')
                    st.markdown(get_image_download_link(result["visualizations"]["keywords_list"], f"{result['topic_name'].replace(' ','_')}_keywords.png"), unsafe_allow_html=True)
                else: st.info("No specific visualizations for this report type or data unavailable.")

            with export_tab:
                st.subheader("Export Report")
                report_file_name = f"{result['topic_name'].replace(' ', '_').lower()}_report_{datetime.now().strftime('%Y%m%d')}.md"
                st.download_button("Download Full Report as Markdown (.md)", result["report_md"], report_file_name, "text/markdown", use_container_width=True)
                st.markdown(f"**File**: `{report_file_name}` | *Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S %Z')}*")
                st.warning("💡 **Remember**: AI-generated. Verify critical data.", icon="❗")

        elif generate_button and not st.session_state.query_input.strip():
            st.error("❗ Please enter a query.", icon="🚨")
    
    with doc_tab:
        st.header("System Information and Documentation")
        st.subheader("1. Approach Explanation"); st.markdown(APPROACH_EXPLANATION_MD, unsafe_allow_html=True)
        st.subheader("2. System Architecture & Workflow"); st.markdown(SYSTEM_DOCUMENTATION_MD, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"© {datetime.now().year} AI Report Generator")

if __name__ == "__main__":
    main()
