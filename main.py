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
# import json # Not actively used, can be removed if not planned for future
# import urllib.parse # Not actively used
from collections import Counter

# Attempt to import duckduckgo_search
try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_SEARCH_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_SEARCH_AVAILABLE = False
    # This warning will be shown in the UI later if needed

# --- Custom NLP Utilities (Replacing NLTK) ---
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
        self.search_results_cache = {}
        self.page_content_cache = {}
        
    def safe_request(self, url, retries=3, timeout=10):
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == retries - 1:
                    st.sidebar.warning(f"Failed to fetch {url} after {retries} attempts: {str(e)[:100]}...")
                    return None
                time.sleep(0.5)
            except Exception as e:
                st.sidebar.warning(f"Unexpected error fetching {url}: {str(e)[:100]}...")
                return None
        return None
    
    def search_web_ddg(self, query, num_results=3):
        if not DUCKDUCKGO_SEARCH_AVAILABLE:
            return self.simulate_search_results(query, num_results)

        cache_key = f"ddg_{query}_{num_results}"
        if cache_key in self.search_results_cache:
            return self.search_results_cache[cache_key]

        try:
            results = []
            with DDGS(timeout=10) as ddgs:
                ddg_results = ddgs.text(query, max_results=num_results)
                for r in ddg_results:
                    results.append({
                        'title': r.get('title', ''),
                        'link': r.get('href', ''),
                        'snippet': r.get('body', '')
                    })
            if not results:
                st.sidebar.info(f"DuckDuckGo search for '{query}' yielded no results. Simulating.")
                return self.simulate_search_results(query, num_results)
            
            self.search_results_cache[cache_key] = results
            return results
        except Exception as e:
            st.sidebar.error(f"DDG search error: {e}. Simulating.")
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
                soup = BeautifulSoup(response.content, 'html.parser')
                for script_or_style in soup(["script", "style", "nav", "footer", "aside", "header", "form", "button", "iframe", "noscript"]): 
                    script_or_style.decompose()
                
                text_parts = []
                # Prioritize common content containers
                main_content = soup.find('main') or \
                               soup.find('article') or \
                               soup.find('div', role='main') or \
                               soup.find('div', class_=re.compile(r'(content|main|body|post|entry)', re.I))
                
                content_container = main_content if main_content else soup.body 

                if content_container:
                    # Extract from meaningful tags first
                    for element in content_container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'div', 'span']):
                        if element.name == 'p' and len(element.get_text(strip=True)) > 50 : 
                             text_parts.append(element.get_text(separator=" ", strip=True))
                        elif element.name in ['h1','h2','h3','h4']:
                             text_parts.append(element.get_text(separator=" ", strip=True) + ".") # Add punctuation for titles
                        elif element.name == 'li' and len(element.get_text(strip=True)) > 10:
                             text_parts.append("- " + element.get_text(separator=" ", strip=True)) # Mark list items
                        # Be more selective with divs and spans
                        elif element.name == 'div' and not element.find(['nav','aside','footer','header','form']) and len(element.get_text(strip=True)) > 100 :
                            # Check if div contains primarily text, not just other complex structures
                            if not element.find(lambda tag: tag.name not in ['p','h1','h2','h3','h4','li','img','figure','table','ul','ol','br','strong','em','a','span'] and tag.string and tag.string.strip()):
                                 text_parts.append(element.get_text(separator=" ", strip=True))
                        elif element.name == 'span' and element.parent.name not in ['button','label','a'] and len(element.get_text(strip=True)) > 30:
                            text_parts.append(element.get_text(separator=" ", strip=True))


                # Fallback if specific tags yield little content
                if not text_parts and content_container: 
                    text_content_full = content_container.get_text(separator='\n', strip=True)
                    text_parts = [line for line in text_content_full.split('\n') if line and len(line) > 20] # Filter short/empty lines

                text_content = "\n".join(text_parts)
                text_content = re.sub(r'\s*\n\s*', '\n', text_content) # Normalize newlines
                text_content = re.sub(r'[ \t]+', ' ', text_content)   # Normalize spaces
                text_content = text_content.strip()

                self.page_content_cache[url] = text_content
                return text_content
            except Exception as e:
                st.sidebar.warning(f"Error parsing {url}: {str(e)[:100]}...")
        return ""
        
    def research_topic_online(self, topic_query, is_industry_query):
        if is_industry_query:
            search_query_detail = "market analysis report size growth key players trends challenges forecast 2023 2024 2025"
        else:
            search_query_detail = "overview summary key points information facts"
        search_query = f"{topic_query} {search_query_detail}"

        st.write(f"🌐 Searching web for: '{search_query}'...")
        search_results_metadata = self.search_web_ddg(search_query, num_results=5) # Increased to 5 for more source material

        combined_text_for_processing = ""
        fetched_sources_list = []

        if not search_results_metadata:
            st.warning("No search results found online. Report will rely on any predefined data or be very limited.")
            market_data_from_web = self.extract_market_data("", topic_query) if is_industry_query else {}
            return market_data_from_web, [], ""

        st.write("📑 Fetching content from top search results (processing up to 3 distinct sources):")
        content_count = 0
        for i, result in enumerate(search_results_metadata): 
            if content_count >= 3: break # Limit to 3 successful fetches for speed
            link = result.get('link')
            title = result.get('title', 'No Title')
            if link:
                st.write(f"   ∟ Attempting fetch: {title} ({link})...")
                content = self.get_content_from_url(link)
                if content and len(content) > 200: # Ensure meaningful content
                    combined_text_for_processing += f"\n\n--- Source: {title} ({link}) ---\n\n" + content + "\n\n" 
                    fetched_sources_list.append({'title': title, 'link': link})
                    content_count += 1
                    st.write(f"     ✅ Content fetched and added from {title}.")
                    time.sleep(0.1) 
                else:
                    st.write(f"     ⚠️ Skipped (empty or too short content) from {link}")
            else:
                st.write(f"  ∟ Skipping result with no link: {title}")

        if not combined_text_for_processing.strip():
             st.warning("Could not fetch detailed content from primary sources. Using search snippets instead.")
             for res in search_results_metadata:
                 if res.get('snippet'):
                     combined_text_for_processing += f"\n\n--- Snippet: {res.get('title')} ---\n\n" + res.get('snippet', '') + "\n\n"
                     if not any(s['link'] == res.get('link') for s in fetched_sources_list) and res.get('link'): # Add as source if not already
                         fetched_sources_list.append({'title': res.get('title', 'Snippet Source'), 'link': res.get('link')})
        
        if not combined_text_for_processing.strip():
            st.warning("No text content gathered from web or snippets. Falling back to fully simulated data if applicable.")
            sim_results = self.simulate_search_results(topic_query, 1)
            if sim_results and sim_results[0].get('snippet'):
                combined_text_for_processing = sim_results[0]['snippet']
                fetched_sources_list.append({'title': sim_results[0]['title'], 'link': sim_results[0]['link']})


        market_data_from_web = {}
        if is_industry_query:
            market_data_from_web = self.extract_market_data(combined_text_for_processing, topic_query) 
    
        return market_data_from_web, fetched_sources_list, combined_text_for_processing.strip()

    def extract_market_data(self, combined_text, industry_topic): 
        market_data = {
            "market_size": "", "cagr": "", "key_players": [], "trends": [], "challenges": [],
            "year": datetime.now().year, "base_year_market_size": "", "forecast_year_market_size": ""
        }
        if not combined_text: return market_data

        # Market Size Extraction (more robust)
        size_patterns = [
            # $1.23 trillion in 2023 ... to $4.56 trillion by 2030
            r"(?:valued at|market size was)\s*(USD|EUR|GBP|¥|\$)\s*([\d,]+\.?\d*)\s*(billion|million|trillion)\s*(?:in|as of)\s*(\d{4}).*?(?:(?:reach|grow to|projected at)\s*(USD|EUR|GBP|¥|\$)\s*([\d,]+\.?\d*)\s*(billion|million|trillion)\s*(?:by|in)\s*(\d{4}))?",
            # $1.23 billion market
            r"(USD|EUR|GBP|¥|\$)\s*([\d,]+\.?\d*)\s*(billion|million|trillion)\s*(?:market|industry)\s*(?:in|as of)?\s*(\d{4})?",
        ]
        found_size = False
        for pattern in size_patterns:
            for match in re.finditer(pattern, combined_text, re.IGNORECASE):
                base_currency, base_value, base_unit, base_year = match.group(1), match.group(2), match.group(3), match.group(4)
                market_data["market_size"] = f"{base_currency.replace('¥','$')}{base_value} {base_unit} ({base_year})" # Normalize currency symbol for display
                market_data["base_year_market_size"] = f"{base_currency.replace('¥','$')}{base_value} {base_unit} ({base_year})"
                
                if match.group(5) and match.group(6) and match.group(7) and match.group(8): # Forecast part
                    fc_currency, fc_value, fc_unit, fc_year = match.group(5), match.group(6), match.group(7), match.group(8)
                    market_data["forecast_year_market_size"] = f"{fc_currency.replace('¥','$')}{fc_value} {fc_unit} ({fc_year})"
                    # Use forecast year for the main market_size if it's more future-oriented
                    market_data["market_size"] = f"{fc_currency.replace('¥','$')}{fc_value} {fc_unit} (Projected {fc_year})"
                found_size = True
                break
            if found_size: break
        
        if not found_size: # Simpler regex if complex one fails
            match = re.search(r"(\$[\d,]+\.?\d*\s*(?:billion|million|trillion))", combined_text, re.IGNORECASE)
            if match: market_data["market_size"] = match.group(1)

        # CAGR Extraction
        cagr_patterns = [
            r"CAGR of\s*([\d.]+\s*%)", r"compound annual growth rate.*?\s*([\d.]+\s*%)",
            r"grow at\s*(?:a CAGR of)?\s*([\d.]+\s*%)", r"expected to grow from.*?at\s*([\d.]+\s*%)",
            r"at a rate of\s*([\d.]+\s*%)",
        ]
        for pattern in cagr_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                market_data["cagr"] = match.group(1).replace(" ", "")
                break
        
        # Key Players Extraction
        companies = []
        player_intro_patterns = [
             r"(?:key players|major players|leading companies|prominent players|market leaders|significant players|key vendors|major vendors|top companies|dominant players)\s*(?:profiled|include|are|such as|comprise|operating in this market are|covered in the report are|analyzed in this report are)\s*:?\s*([^.]+)\.",
             r"The report profiles key players such as\s*(.*?)(?:and other prominent vendors|which include|among others|The study also includes|$)",
             r"Some of the major companies that are present in the market are\s*(.*?)\."
        ]
        raw_player_strings = []
        for pattern in player_intro_patterns:
            for match_obj in re.finditer(pattern, combined_text, re.IGNORECASE): 
                raw_player_strings.append(match_obj.group(1))
        
        # Known company names related to predefined industries (expand this as needed)
        specific_company_patterns_map = {
            "electric vehicle": r"\b(Tesla|BYD|Volkswagen|SAIC|Stellantis|Mercedes-Benz|Ford|General Motors|Hyundai-Kia|Toyota|NIO|XPeng|Li Auto)\b",
            "artificial intelligence": r"\b(Google|Alphabet|Microsoft|Amazon|AWS|NVIDIA|IBM|Meta|OpenAI|Anthropic|Baidu|Apple|Salesforce|Oracle|Intel|AMD|Palantir|C3.ai)\b",
            "renewable energy": r"\b(NextEra Energy|Enel|Iberdrola|EDF|Ørsted|Vestas|First Solar|Canadian Solar|Siemens Gamesa|LONGi|Jinko Solar|Trina Solar|GE Renewable Energy)\b",
            "cloud computing": r"\b(Amazon Web Services|AWS|Microsoft Azure|Azure|Google Cloud Platform|GCP|Alibaba Cloud|IBM Cloud|Oracle Cloud|Salesforce|SAP|VMware|Rackspace|DigitalOcean)\b"
        }
        
        # General company name patterns (ends with Corp, Inc, LLC, Ltd, etc.)
        generic_company_pattern = r"\b([A-Z][\w\s&-]+(?:Inc\.|LLC|Corp\.|Ltd\.|GmbH|S\.A\.))\b"
        companies.extend(re.findall(generic_company_pattern, combined_text))

        # Add specific companies if industry topic matches
        for industry_keyword, pattern in specific_company_patterns_map.items():
            if industry_keyword in industry_topic.lower():
                matches = re.findall(pattern, combined_text)
                companies.extend(m for m in matches if m and m not in companies)

        for player_list_str in raw_player_strings:
            # Split by common delimiters, handle 'and' carefully
            potential_players = re.split(r',\s*(?:and\s+)?|\s+and\s+|;\s*|\s*&\s*', player_list_str)
            for player in potential_players:
                player = player.strip()
                # Filter out generic terms, ensure it starts with a capital, reasonable length
                if player and len(player.split()) <= 5 and player[0].isupper() \
                   and not player.lower().endswith(("etc.", "e.g.", "others")) \
                   and not player.lower() in ["various", "many", "several", "leading", "major", "key"] \
                   and len(player) > 2 and len(player) < 50:
                    if player not in companies: 
                        companies.append(player)
        
        seen = set()
        # Prioritize longer, more specific names if there are shorter versions (e.g., "Google" vs "Google Inc.")
        companies.sort(key=len, reverse=True) 
        unique_players = []
        for player in companies:
            is_substring = False
            for seen_player in seen:
                if player in seen_player:
                    is_substring = True
                    break
            if not is_substring:
                unique_players.append(player)
                seen.add(player)
        market_data["key_players"] = unique_players[:15] # Increased limit for key players

        # Trends and Challenges Extraction (using custom sentence tokenization)
        try:
            sentences = custom_sent_tokenize(combined_text)
            
            trend_keywords = ["innovation", "advancement", "growth in", "adoption of", "increasing demand", 
                              "shift towards", "emergence of", "rising popularity", "expanding use", "key trend", "driving factor", "opportunity", "development"]
            trends_found = [s.strip() for s in sentences if any(kw in s.lower() for kw in trend_keywords) and "market" in s.lower() and 30 < len(s.strip()) < 300]
            # Further filter trends to be more relevant
            filtered_trends = []
            for t in trends_found:
                if "challenge" not in t.lower() and "obstacle" not in t.lower() and "risk" not in t.lower():
                     # Simple check for novelty or action verb
                    if re.search(r"\b(is|are|will be|expected to|driving|leading to|enabling)\b", t.lower()):
                        filtered_trends.append(t)
            market_data["trends"] = list(dict.fromkeys(filtered_trends))[:7] # More trends

            challenge_keywords = ["challenge", "obstacle", "barrier", "issue", "concern", "risk", "limitation", "constraint", "difficulty", "threat", "hindrance", "restraint"]
            challenges_found = [s.strip() for s in sentences if any(kw in s.lower() for kw in challenge_keywords) and "market" in s.lower() and 30 < len(s.strip()) < 300]
            market_data["challenges"] = list(dict.fromkeys(challenges_found))[:5] # More challenges

        except Exception as e:
            st.sidebar.warning(f"Market data text processing error: {e}")
            if not market_data.get("trends"): market_data["trends"] = ["Trend extraction encountered an issue."]
            if not market_data.get("challenges"): market_data["challenges"] = ["Challenge extraction encountered an issue."]
        return market_data

    def extract_general_summary_and_keywords(self, combined_text, query_text, num_summary_sentences=7, num_keywords=10):
        summary_data = {
            "summary_sentences": [], "keywords": [],
            "full_text_snippet": combined_text[:3000] + "..." if len(combined_text) > 3000 else combined_text
        }
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
                if not (30 < len(s) < 500): continue # Filter by length first

                score = 0
                sent_words = custom_word_tokenize(s)
                
                # Score based on query word overlap
                for qw in query_content_words:
                    if qw in sent_words: score += 2 # Higher weight for query words
                
                # Score based on position (higher score for earlier sentences)
                score += max(0, (10 - i) * 0.1) # Small bonus for being early

                # Score based on TF-IDF-like term frequency (simple version)
                # (Not a full TF-IDF, but rewards sentences with less common, important words)
                # For simplicity, we'll just reward non-stop words.
                num_content_words = len([w for w in sent_words if w not in stop_words and len(w) > 3])
                score += num_content_words * 0.05

                # Penalize if too generic or too many stop words
                if len(sent_words) > 0 and (len(sent_words) - num_content_words) / len(sent_words) > 0.7:
                    score *= 0.5

                sentence_scores.append((s, score, i)) 

            # Sort by score (descending), then by original position (ascending)
            sentence_scores.sort(key=lambda x: (-x[1], x[2])) 
            
            # Select top sentences, ensuring some diversity if possible (not implemented here for simplicity)
            selected_scored_sentences = [s_info[0] for s_info in sentence_scores]
            
            # Ensure uniqueness while preserving order of first appearance among high-scorers
            summary_data["summary_sentences"] = list(dict.fromkeys(selected_scored_sentences))[:num_summary_sentences]


            words = custom_word_tokenize(combined_text)
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 3]
            
            if filtered_words:
                word_counts = Counter(filtered_words)
                # Boost keywords that are also in the query
                for qw in query_content_words:
                    if qw in word_counts:
                        word_counts[qw] *= 1.5 
                summary_data["keywords"] = [kw for kw, count in word_counts.most_common(num_keywords)]
        except Exception as e:
            st.sidebar.warning(f"General text processing error: {e}")
            if not summary_data.get("summary_sentences"):
                summary_data["summary_sentences"] = [combined_text[:500] + "..."] if combined_text else ["Could not process text."]
        return summary_data

    def simulate_search_results(self, query, num_results=3):
        query_lower = query.lower()
        results = []
        generic_results = [
            {'title': f'{query.capitalize()} General Overview', 'link': f'#simulated/{query_lower.replace(" ","-")}-overview', 
             'snippet': f'This is a simulated general overview of {query_lower}. It typically covers key aspects, definitions, and related information. For instance, if {query_lower} is a market, it might touch upon its size (e.g., USD 100 Billion in 2023) and growth rate (e.g., CAGR 10%). Key entities often mentioned include AlphaOrg and BetaCorp.'},
            {'title': f'Detailed Insights into {query.capitalize()}', 'link': f'#simulated/{query_lower.replace(" ","-")}-details', 
             'snippet': f'Further simulated details on {query_lower}, including discussions on its impact, relevance, and potential trends. Challenges could involve market saturation or technological disruption. Opportunities might arise from innovation.'},
            {'title': f'{query.capitalize()} Key Considerations & Players', 'link': f'#simulated/{query_lower.replace(" ","-")}-points', 
             'snippet': f'Important simulated points regarding {query_lower}. Some notable entities simulated are Gamma Inc., Delta LLC. These players might have significant market share.'}
        ]
        if "electric vehicle" in query_lower: results = [{'title': 'Simulated EV Market Overview', 'link': '#simulated/ev-market', 'snippet': 'Electric vehicles (EVs) are gaining traction globally. Simulated key players: Tesla, BYD. Simulated market size: USD 500 Billion (2023), with a CAGR of around 18%. Trends include battery tech advancements and charging infrastructure expansion.'}]
        elif "ai market" in query_lower or "artificial intelligence" in query_lower: results = [{'title': 'Simulated AI Market Summary', 'link': '#simulated/ai-market', 'snippet': 'Artificial intelligence (AI) is a rapidly evolving field. Simulated leaders: Google, Microsoft. Simulated CAGR: ~35%. Trends point to generative AI and cross-industry adoption.'}]
        else: results = generic_results
        return results[:num_results]

class QueryTopicResearcher: 
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
        self.predefined_industry_data = { # Simplified, more data would be in a DB or external files
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
            # Add more predefined industries here for richer "known" reports
        }

    def identify_query_topic_and_type(self, query_text):
        query_lower = query_text.lower()
        for industry_key, keywords_list in self.known_industries_keywords_map.items():
            for keyword in keywords_list:
                if keyword in query_lower:
                    # Check for "report", "analysis", "trends", "market" to confirm industry_analysis intent
                    if any(term in query_lower for term in ["report", "analysis", "trends", "market", "industry", "competitors", "outlook"]):
                        return industry_key, "industry_analysis"
        
        general_industry_terms = ["market", "industry", "sector", "cagr", "market share", "competitors", "trends analysis", "strategic report"]
        if any(term in query_lower for term in general_industry_terms):
            # Try to extract the specific industry name if it's a generic industry query
            # e.g., "report on the footwear market"
            match = re.search(r"(?:for|of|on|about|analyze|investigate|report on)\s+(?:the\s+)?([\w\s\-]+?)\s+(?:market|industry|sector)", query_lower)
            if match:
                extracted_topic = match.group(1).strip()
                # Check if this extracted topic matches one of our known industries again
                for industry_key_refined, keywords_list_refined in self.known_industries_keywords_map.items():
                    if extracted_topic == industry_key_refined or extracted_topic in keywords_list_refined:
                         return industry_key_refined, "industry_analysis"
                return extracted_topic, "industry_analysis" # Treat as new industry

        # Default to general information if no strong industry signals
        topic = re.sub(r"^(?:what is|tell me about|generate a report on|analyze|information on|provide details on|summarize)\s*(?:the\s+)?", "", query_lower, flags=re.IGNORECASE).strip()
        topic = topic.replace("?", "").replace(" report", "").replace(" analysis", "") # Clean up common suffixes
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
                st.info(f"'{identified_topic.title()}' is being analyzed as an industry. No specific pre-configuration found. Report will heavily rely on web search and generic industry templates. Quality may vary.")
                final_data_payload = { # Default structure for unknown industries
                    "market_size": "", "cagr": "", "key_players": [], "trends": [], "challenges": [],
                    "market_share": {}, "regions": {}, "segments": {}, "forecast": {}, "opportunities": [],
                    "year": datetime.now().year,
                    "base_year_market_size": "", "forecast_year_market_size": ""
                }
            
            # Merge web data with predefined data, web data takes precedence for dynamic fields
            if web_specific_market_data.get("market_size"): final_data_payload["market_size"] = web_specific_market_data["market_size"]
            if web_specific_market_data.get("base_year_market_size"): final_data_payload["base_year_market_size"] = web_specific_market_data["base_year_market_size"]
            if web_specific_market_data.get("forecast_year_market_size"): final_data_payload["forecast_year_market_size"] = web_specific_market_data["forecast_year_market_size"]
            if web_specific_market_data.get("cagr"): final_data_payload["cagr"] = web_specific_market_data["cagr"]
            
            # For players, trends, challenges: combine and de-duplicate, prioritizing web if available
            for field in ["key_players", "trends", "challenges"]:
                web_values = web_specific_market_data.get(field, [])
                predefined_values = final_data_payload.get(field, [])
                combined_values = web_values + [pv for pv in predefined_values if pv not in web_values]
                final_data_payload[field] = list(dict.fromkeys(combined_values))[:10 if field == "key_players" else (7 if field == "trends" else 5)]


            # Ensure essential fields have fallbacks if still empty
            final_data_payload.setdefault("key_players", ["Undetermined Key Players from Web Search"])
            final_data_payload.setdefault("trends", ["General industry developments are typically observed based on web data."])
            final_data_payload.setdefault("challenges", ["Standard competitive pressures and operational hurdles are common."])
            final_data_payload.setdefault("opportunities", ["Exploration of new market segments post-analysis.", "Technological integration for efficiency."])
            final_data_payload.setdefault("market_share", {"Top Player (Est. from Web/Generic)": 30, "Challenger (Est.)": 20, "Other Competitors": 50})
            final_data_payload.setdefault("regions", {"Primary Region (Global/Dominant)": 60, "Secondary Region": 30, "Other Regions": 10})
            final_data_payload.setdefault("segments", {"Main Product/Service Segment": 70, "Other Segments": 30})
            final_data_payload.setdefault("forecast", {"Overall Market Trend (Illustrative)": [100, 110, 121]}) # Placeholder growth

        elif query_type == "general_information":
            general_summary_data = self.web_researcher.extract_general_summary_and_keywords(combined_web_text, query_text_input)
            final_data_payload = general_summary_data
            final_data_payload['raw_text_sample'] = combined_web_text[:2000] # Keep sample for potential display

        final_data_payload["fetched_sources"] = fetched_sources
        final_data_payload["query_type"] = query_type
        final_data_payload["query_topic"] = identified_topic 
        final_data_payload["original_query"] = query_text_input
        final_data_payload["raw_web_text_for_analysis"] = combined_web_text # For deeper analysis if needed
        
        return final_data_payload

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
        
        # Filter out "Others" and ensure values are numeric for calculations
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
            "market_share_data": market_share_data, # Keep original for charting
            "market_concentration": {"cr4": cr4, "structure": structure},
            "leader_advantage_points": (sorted_shares[0][1] - sorted_shares[1][1]) if len(sorted_shares) > 1 else (sorted_shares[0][1] if sorted_shares else 0),
            "top_players_list": [name for name, _ in sorted_shares[:5]] if sorted_shares else (key_players[:3] if key_players and key_players[0] != "Undetermined Key Players from Web Search" else ["N/A"])
        }
        return analysis
        
    def analyze_regional_impact(self, industry_data):
        regions_data = industry_data.get("regions", {"Global Focus (Undetermined Detail)": 100})
        dominant_region_tuple = max(regions_data.items(), key=lambda x: x[1]) if regions_data and any(regions_data.values()) else ("N/A", 0)
        dominant_region = dominant_region_tuple[0]
        
        cagr_str = str(industry_data.get("cagr","5%")).replace('%','').replace('N/A','5')
        try:
            cagr_val = float(cagr_str)
        except ValueError:
            cagr_val = 5.0 # Default if conversion fails

        regional_growth_rates = {r: round(random.uniform(max(1, cagr_val * 0.7), cagr_val * 1.3), 1) for r in regions_data.keys()}
        fastest_growing_region_tuple = max(regional_growth_rates.items(), key=lambda x: x[1]) if regional_growth_rates else ("N/A", 0)
        fastest_growing_region = fastest_growing_region_tuple[0]
        
        emerging_markets_list = [
            r for r,s_val in regions_data.items() 
            if isinstance(s_val, (int,float)) and s_val < 20 and regional_growth_rates.get(r,0) > (cagr_val * 0.9)
        ]

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
        dominant_segment_tuple = max(segments_data.items(), key=lambda x: x[1]) if segments_data and any(segments_data.values()) else ("N/A", 0)
        dominant_segment = dominant_segment_tuple[0]

        cagr_str = str(industry_data.get("cagr","5%")).replace('%','').replace('N/A','5')
        try:
            cagr_val = float(cagr_str)
        except ValueError:
            cagr_val = 5.0

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
        forecast_data = industry_data.get("forecast", {"Overall Market Index": [100,110,121,133]}) # Default if none
        if not isinstance(forecast_data, dict) or not forecast_data: 
            forecast_data = {"Overall Market Index": [100,110,121,133]}
        
        # Ensure forecast series are valid lists of numbers
        for k,v_list in forecast_data.items(): 
            if not isinstance(v_list, list) or not all(isinstance(x,(int,float)) for x in v_list) or len(v_list) < 2:
                # Generate a plausible random series if data is bad
                start_val = random.randint(50,150)
                forecast_data[k] = [round(start_val * (1 + random.uniform(-0.05, 0.15))**i) for i in range(random.randint(3,5))]


        challenges_list = industry_data.get("challenges", ["Generic challenge: Economic uncertainty.", "Generic challenge: Adapting to new technologies."])
        opportunities_list = industry_data.get("opportunities", ["Generic opportunity: Market expansion into new demographics.", "Generic opportunity: Leveraging data analytics for insights."])
        
        trend_analysis_map = {}
        for co_name, data_series in forecast_data.items():
            if len(data_series)>1:
                abs_trend = data_series[-1] - data_series[0]
                percent_change = round((abs_trend) / data_series[0] * 100, 1) if data_series[0] != 0 else 0
                direction = "increasing" if abs_trend > 0 else ("decreasing" if abs_trend < 0 else "stable")
                trend_analysis_map[co_name] = {
                    "data_series": data_series, "absolute_change": abs_trend, 
                    "direction": direction, "percentage_change": percent_change
                }
        
        # Identify fastest growing entities from forecast if available
        growing_entities = sorted(
            [(k,v["percentage_change"]) for k,v in trend_analysis_map.items() if v["percentage_change"] > 0 and k.lower() != "others" and "overall" not in k.lower()],
            key=lambda x:x[1], reverse=True
        )
        
        analysis = {
            "forecast_chart_data": forecast_data, 
            "forecast_trend_analysis": trend_analysis_map,
            "fastest_growing_forecasted_entities": [name for name,_ in growing_entities[:3]] if growing_entities else ["N/A - Detailed player forecasts not available or all are stable/declining."],
            "key_challenges_outlook": challenges_list[:3], # Top 3 challenges
            "key_opportunities_outlook": opportunities_list[:3] # Top 3 opportunities
        }
        return analysis

class ReportGenerator:
    def generate_executive_summary(self, topic_name, market_analysis, competitor_analysis, future_analysis):
        market_size = market_analysis.get("market_size", "N/A")
        growth_rate = market_analysis.get("growth_rate", "N/A")
        
        top_players_list = competitor_analysis.get("top_players_list", ["Key industry participants"])
        top_players_str = ", ".join(top_players_list[:3]) if top_players_list and top_players_list[0] != "N/A" else "leading industry participants"
        
        market_structure = competitor_analysis.get("market_concentration", {}).get("structure", "competitive")
        cr4_value = competitor_analysis.get("market_concentration", {}).get("cr4", 0)
        cr4_text = f"with the top four players accounting for approximately {cr4_value:.1f}% of the market share" if cr4_value > 0 else "indicating a potentially fragmented or data-limited landscape"

        key_trends_list = market_analysis.get("key_trends", ["general industry developments", "further market evolution"])
        trend1 = key_trends_list[0].lower().strip('.') if key_trends_list else "general industry developments"
        
        key_opportunities_list = future_analysis.get("key_opportunities_outlook", ["strategic growth areas"])
        opportunity1 = key_opportunities_list[0].lower().strip('.') if key_opportunities_list else "strategic growth areas"

        return f"""## Executive Summary
The **{topic_name.title()}** industry presents a dynamic landscape. Currently valued at approximately **{market_size}**, the market is projected to expand at a Compound Annual Growth Rate (CAGR) of around **{growth_rate}**. 

Key strategic insights include:
1.  **Market Dynamics & Structure**: The market is characterized as **{market_structure}**, {cr4_text}.
2.  **Competitive Landscape**: Dominant players include **{top_players_str}**, who are shaping industry benchmarks and competitive responses.
3.  **Primary Growth Drivers**: A significant trend influencing growth is **{trend1}**.
4.  **Strategic Opportunities**: Key opportunities for stakeholders lie in areas such as **{opportunity1}**.

This report provides a detailed analysis of these factors, offering data-driven insights and strategic recommendations for navigating and succeeding in the {topic_name.title()} market.
"""

    def generate_market_overview(self, topic_name, market_analysis_data, regional_analysis_data, segment_analysis_data):
        ma = market_analysis_data; ra = regional_analysis_data; sa = segment_analysis_data
        
        market_size_text = ma.get("market_size","N/A")
        growth_rate_text = ma.get("growth_rate","N/A")
        
        # Regional Analysis
        dominant_region = ra.get("dominant_region_name","N/A")
        dominant_region_share = ra.get("dominant_region_share","N/A")
        fastest_growing_region = ra.get("fastest_growing_region_name","N/A")
        fastest_growing_rate_region = ra.get("fastest_growing_region_rate","N/A")
        regional_dist_text = f"{dominant_region} currently leads the market, holding approximately {dominant_region_share}% share. The fastest-growing region is anticipated to be {fastest_growing_region} (estimated growth: {fastest_growing_rate_region}% CAGR)."
        if dominant_region == "N/A": regional_dist_text = "Detailed regional distribution data is limited; a global or broad regional focus is assumed."

        # Segment Analysis
        dominant_segment = sa.get("dominant_segment_name","N/A")
        dominant_segment_share = sa.get("dominant_segment_share","N/A")
        fastest_growing_segment = sa.get("fastest_growing_segment_name","N/A")
        fastest_growing_rate_segment = sa.get("fastest_growing_segment_rate","N/A")
        segment_dist_text = f"The market is segmented by various categories, with {dominant_segment} being the dominant segment, capturing about {dominant_segment_share}% of the market. {fastest_growing_segment} is projected as the fastest-growing segment (estimated growth: {fastest_growing_rate_segment}% CAGR)."
        if dominant_segment == "N/A": segment_dist_text = "Specific market segmentation data is limited. The market is likely composed of several product/service categories."
        
        # Market Drivers (from trends)
        key_trends = ma.get("key_trends", [])
        drivers_text = ""
        if key_trends:
            drivers_text = "Key market drivers include:\n"
            for i, trend in enumerate(key_trends[:3]): # Display top 3 trends as drivers
                drivers_text += f"    - {trend.strip('.')}\n"
        else:
            drivers_text = "Market drivers are evolving, typically influenced by technological advancements, consumer demand shifts, and economic factors."

        return f"""## Market Overview
### Market Size and Growth Projections
The global **{topic_name.title()}** market is currently estimated at **{market_size_text}**. It is anticipated to experience a Compound Annual Growth Rate (CAGR) of approximately **{growth_rate_text}** over the forecast period. This growth trajectory reflects ongoing industry developments and increasing demand.

### Regional Market Insights
{regional_dist_text}

### Key Market Segments
{segment_dist_text}
The distribution across segments like {", ".join(list(sa.get("segment_distribution_data",{}).keys())) if sa.get("segment_distribution_data",{}) else 'various categories'} highlights diverse opportunities and specific growth areas.

### Primary Market Drivers
{drivers_text}
"""

    def generate_competitor_analysis(self, topic_name, competitor_analysis_data):
        ca = competitor_analysis_data
        mc = ca.get("market_concentration",{})
        market_structure = mc.get("structure","competitive")
        cr4_value = mc.get("cr4",0)
        cr4_desc = f"The top four firms collectively hold approximately {cr4_value:.1f}% of the market share." if cr4_value > 0 else "Market share concentration data for top firms is limited, suggesting a potentially fragmented landscape or need for deeper primary research."

        key_players_list = ca.get("key_players", [])
        players_intro = f"The competitive landscape of the {topic_name.title()} market features several key players. Prominent companies identified include:"
        if not key_players_list or key_players_list == ["Undetermined Key Players from Web Search"]:
             players_intro = "Specific key player identification was limited in the automated search. The market likely consists of established entities and emerging innovators."
             players_list_md = "- Further investigation is needed to detail specific competitor profiles."
        else:
            players_list_md = "\n".join([f"- **{player.strip()}**" for player in key_players_list[:8]]) # List top 8

        leader_advantage = ca.get("leader_advantage_points",0)
        dynamics_intensity = "intense, with narrow gaps between leading firms" if 0 < leader_advantage < 10 else "dynamic, with notable differentiation among top players"
        if leader_advantage == 0 and cr4_value == 0: dynamics_intensity = "likely diverse with numerous participants"
        
        comp_dynamics_text = f"""The competitive dynamics are **{dynamics_intensity}**. Strategic positioning, innovation, and market reach are critical differentiators.
The leader's advantage over the immediate challenger is approximately {leader_advantage:.1f} market share points, based on available data.
Market participants compete on factors such as product/service innovation, pricing strategies, brand reputation, and customer service."""
        if leader_advantage == 0 and cr4_value == 0 and not key_players_list :
             comp_dynamics_text = "Competitive dynamics are presumed to be active, driven by innovation and customer acquisition efforts. Specific intensity levels require more granular data."


        return f"""## Competitor Analysis
### Market Concentration and Structure
The **{topic_name.title()}** market is characterized as **{market_structure}**. {cr4_desc} This structure influences competitive strategies and market entry barriers.

### Profile of Key Players
{players_intro}
{players_list_md}
*(Note: This list is based on available public information and may not be exhaustive.)*

### Competitive Dynamics & Positioning
{comp_dynamics_text}
Companies are continuously adapting to evolving consumer preferences and technological advancements to maintain or enhance their market positions.
"""

    def generate_trends_analysis(self, topic_name, market_analysis_data):
        trends = market_analysis_data.get("key_trends", [])
        strengths = market_analysis_data.get("trend_strength", []) # Simulated
        
        analysis_intro = f"## Key Industry Trends & Developments\nSeveral influential trends are shaping the future trajectory of the **{topic_name.title()}** market. Understanding these trends is crucial for strategic planning and identifying growth avenues."
        
        if not trends or trends == ["No specific trends identified from available data."]:
            return analysis_intro + "\n\nDetailed trend analysis based on automated web research was limited. General industry evolution often includes:\n- Technological advancements and digitalization.\n- Shifting consumer behaviors and preferences.\n- Increased focus on sustainability and ESG factors.\n- Regulatory changes impacting operations."

        trends_md = ""
        for i, trend_desc in enumerate(trends[:5]): # Show top 5 trends
            strength_val = strengths[i] if i < len(strengths) else random.randint(70,90)
            impact_level = "High" if strength_val > 85 else ("Medium" if strength_val > 70 else "Moderate")
            
            trends_md += f"\n### {i+1}. {trend_desc.strip('.')}\n"
            trends_md += f"   - **Significance**: This trend is considered to have a **{impact_level.lower()} impact** on the market (estimated influence: {strength_val}/100).\n"
            trends_md += f"   - **Implication**: Businesses need to adapt to this by [e.g., investing in relevant technologies, revising strategies, or exploring new models related to this trend].\n" # Generic placeholder for implication
        
        return analysis_intro + trends_md

    def generate_strategic_recommendations(self, topic_name, market_analysis, competitor_analysis, future_analysis):
        fa = future_analysis; ca = competitor_analysis; ma = market_analysis
        
        opportunities = fa.get("key_opportunities_outlook", ["new technological integrations", "expansion into underserved customer segments"])[:2]
        challenges = fa.get("key_challenges_outlook",["economic volatility", "increasing regulatory scrutiny"])[:2]
        
        cr4 = ca.get("market_concentration",{}).get("cr4",50)
        market_structure = ca.get("market_concentration",{}).get("structure","competitive")

        rec_list = []

        # Recommendation 1: Market Positioning
        if "concentrated" in market_structure.lower() or cr4 > 60:
            rec_list.append(f"**Market Positioning**: In a {market_structure.lower()} market, focus on differentiation strategies, niche market penetration, or strategic alliances to compete effectively against dominant players. Emphasize unique value propositions.")
        elif "fragmented" in market_structure.lower() or (cr4 < 40 and cr4 > 0):
            rec_list.append(f"**Market Positioning**: In a more {market_structure.lower()} landscape, pursue market share growth through aggressive innovation, customer acquisition, and potentially consolidation strategies (M&A).")
        else: # Moderately competitive or unknown
            rec_list.append(f"**Market Positioning**: Continuously assess the competitive landscape. Focus on building strong brand equity and customer loyalty. Adaptability to market shifts is key.")

        # Recommendation 2: Leverage Opportunities
        op1 = opportunities[0].lower().strip('.') if opportunities else "identified growth areas"
        rec_list.append(f"**Leverage Key Opportunities**: Prioritize investment and strategic focus on emerging opportunities such as **{op1}**. Develop roadmaps to capitalize on these growth drivers.")

        # Recommendation 3: Mitigate Challenges
        ch1 = challenges[0].lower().strip('.') if challenges else "prevailing industry hurdles"
        rec_list.append(f"**Address Key Challenges**: Proactively develop strategies to mitigate risks associated with **{ch1}**. This may involve diversifying supply chains, enhancing operational resilience, or lobbying efforts.")

        # Recommendation 4: Innovation & Technology
        tech_driven = False
        if ma.get("key_trends"):
            for trend in ma.get("key_trends"):
                if any(t_kw in trend.lower() for t_kw in ["technology", "digital", "ai", "automation", "innovation"]):
                    tech_driven = True
                    break
        if tech_driven:
             rec_list.append(f"**Foster Innovation & Technology Adoption**: Given the influence of technological advancements (as noted in market trends), accelerate R&D and adoption of relevant technologies to enhance product/service offerings and operational efficiency.")
        else:
             rec_list.append(f"**Drive Continuous Improvement**: Focus on incremental innovation and operational excellence to maintain a competitive edge, even if disruptive tech is not the primary driver.")
        
        # Recommendation 5: Customer Centricity
        rec_list.append(f"**Enhance Customer Centricity**: Deepen understanding of customer needs and preferences within the {topic_name.title()} market. Invest in customer experience (CX) and personalized offerings to build loyalty and drive retention.")

        # Recommendation 6: Strategic Partnerships
        rec_list.append(f"**Explore Strategic Alliances**: Consider collaborations, joint ventures, or partnerships to access new markets/technologies, share risks, or enhance competitive capabilities, particularly if addressing complex challenges like {challenges[1].lower().strip('.') if len(challenges)>1 else 'market entry barriers'}.")

        recommendations_md = "\n".join([f"{i+1}. {rec}" for i, rec in enumerate(rec_list)])

        return f"""## Strategic Recommendations
Based on the analysis of the **{topic_name.title()}** market, the following strategic recommendations are proposed for stakeholders aiming to achieve sustained growth and competitive advantage:

{recommendations_md}

*Disclaimer: These recommendations are high-level and derived from automated analysis. They should be validated with in-depth primary research and expert consultation before implementation.*
"""

    def generate_future_outlook(self, topic_name, future_analysis_data):
        fa = future_analysis_data
        
        fast_growing_players = fa.get("fastest_growing_forecasted_entities",[])
        if not fast_growing_players or fast_growing_players[0].startswith("N/A"):
            outlook_dynamics = "The market is expected to continue its current trajectory, with established leaders likely maintaining their positions while innovative smaller players seek to disrupt."
        else:
            outlook_dynamics = f"Future market dynamics may see increased competition from agile players like {', '.join(fast_growing_players)}, potentially challenging established leaders through innovation and niche strategies."

        key_challenges_outlook = fa.get("key_challenges_outlook", [])
        challenges_text = "Key challenges that will continue to shape the market include:\n" + \
                          "\n".join([f"    - {ch.strip('.')}" for ch in key_challenges_outlook]) if key_challenges_outlook else \
                          "Ongoing adaptation to economic shifts and technological disruptions will remain critical."

        key_opportunities_outlook = fa.get("key_opportunities_outlook", [])
        opportunities_text = "Significant long-term opportunities are anticipated in areas such as:\n" + \
                             "\n".join([f"    - {op.strip('.')}" for op in key_opportunities_outlook]) if key_opportunities_outlook else \
                             "Exploiting technological advancements and evolving consumer needs will unlock new growth avenues."

        # Critical Uncertainties: derived from challenges or generic
        uncertainty1 = key_challenges_outlook[0].lower().strip('.') if key_challenges_outlook else "the pace of technological change"
        uncertainty2 = key_challenges_outlook[1].lower().strip('.') if len(key_challenges_outlook) > 1 else "global economic conditions and regulatory shifts"


        return f"""## Future Market Outlook
### Anticipated Market Evolution (Next 3-5 Years)
The **{topic_name.title()}** market is poised for continued evolution over the next 3-5 years. 
{outlook_dynamics}
Factors such as sustainability, digitalization, and changing consumer expectations will likely play a more significant role.

### Key Opportunities on the Horizon
{opportunities_text}
Early identification and strategic investment in these areas can yield substantial competitive advantages.

### Persistent and Emerging Challenges
{challenges_text}
Navigating these challenges effectively will be crucial for sustained success.

### Critical Uncertainties
The future landscape is subject to critical uncertainties, including:
1.  **Impact of {uncertainty1.capitalize()}**: How this evolves will significantly affect market strategies and investment.
2.  **Influence of {uncertainty2.capitalize()}**: These external factors could introduce both risks and unforeseen opportunities.

Proactive monitoring and scenario planning are recommended to navigate these uncertainties.
"""

    def generate_general_introduction(self, original_query, topic_name):
        return f"""## Introduction
This report provides a summary of information related to your query: "{original_query}". The research focuses on the topic of **{topic_name.title()}** based on publicly available web content gathered at the time of generation.
"""

    def generate_general_summary(self, summary_sentences):
        if not summary_sentences or not isinstance(summary_sentences, list) or not summary_sentences[0]:
            return "## Key Information Summary\nNo specific summary points could be reliably extracted from the web content for this query.\n"
        
        summary_points = "\n".join([f"- {s.strip()}" for s in summary_sentences])
        return f"""## Key Information Summary
Based on the automated web research, the following key points or summary sentences were extracted that appear relevant to **{st.session_state.current_topic_name.title()}**:
{summary_points}"""

    def generate_general_keywords(self, keywords):
        if not keywords or not isinstance(keywords, list) or not keywords[0]:
            return "\n## Main Keywords/Topics Identified\nNo distinct keywords were prominently identified from the processed content.\n"
            
        keywords_list_md = "\n".join([f"- {kw.capitalize()}" for kw in keywords])
        return f"""## Main Keywords/Topics Identified
The research identified the following as prominent keywords or topics within the fetched content related to **{st.session_state.current_topic_name.title()}**:
{keywords_list_md}"""
    
    def generate_text_snippet_section(self, text_snippet): # Optional, can make report very long
        if not text_snippet or len(text_snippet) < 100: return "" # Only show if substantial
        return f"""\n## Extended Content Snippet
Below is an aggregated snippet from the web content that formed the basis of this summary:
\n---\n{text_snippet}\n---"""

    def compile_full_report(self, topic_name, sections, fetched_sources=None, query_type="industry_analysis"):
        current_date = datetime.now().strftime("%B %d, %Y")
        
        report_title_main = f"{topic_name.title()} Information Report"
        report_subtitle = "Key Insights and Summary from Automated Web Research"
        toc_items_map = {
            "Introduction": "introduction",
            "Key Information Summary": "key-information-summary",
            "Main Keywords/Topics Identified": "main-keywords-topics-identified"
        }

        if query_type == "industry_analysis":
            report_title_main = f"{topic_name.title()} Industry Intelligence Report"
            report_subtitle = "Comprehensive Market Analysis and Strategic Recommendations (AI-Generated)"
            toc_items_map = {
                "Executive Summary": "executive-summary",
                "Market Overview": "market-overview",
                "Competitor Analysis": "competitor-analysis",
                "Key Industry Trends & Developments": "key-industry-trends-developments", # Matched section title
                "Strategic Recommendations": "strategic-recommendations",
                "Future Market Outlook": "future-market-outlook"
            }

        title_md = f"# {report_title_main}\n### {report_subtitle}\n*Report Generated: {current_date}*\n"
        
        toc_md = "\n## Table of Contents\n"
        for i, (item_name, anchor_base) in enumerate(toc_items_map.items()):
            # anchor = item_name.lower().replace(" ", "-").replace("/", "").replace("&","") # Simpler anchor
            toc_md += f"{i+1}. [{item_name}](#{anchor_base})\n"

        appendix_toc_num = len(toc_items_map) + 1
        if fetched_sources:
            toc_md += f"{appendix_toc_num}. [Data Sources Appendix](#data-sources-appendix)\n"

        full_report = title_md + toc_md + "".join(sections)
        
        if fetched_sources:
            sources_appendix = f"\n## Data Sources Appendix\nThis report utilized information synthesized from the following publicly accessible web pages (accessed on {current_date}):\n"
            for i, source in enumerate(fetched_sources):
                title = source.get('title','Untitled Source')
                link = source.get('link','#error-no-link')
                # Sanitize title for markdown if it contains problematic characters for links
                title_sanitized = re.sub(r'[\[\]]', '', title) 
                sources_appendix += f"{i+1}. **[{title_sanitized}]({link})**\n"
            full_report += sources_appendix
        
        disclaimer = """\n\n---\n**Disclaimer:** *This report is generated by an automated AI system based on information retrieved from public web sources and, where applicable, predefined datasets. While efforts are made to ensure accuracy, the information is provided "as-is" without warranties of any kind. This report is intended for informational and preliminary research purposes only and should not be the sole basis for business, financial, or strategic decisions. Users are advised to conduct their own thorough due diligence and consult with qualified professionals before making any critical decisions. The content reflects information available at the time of generation and may not include the most current developments.*"""
        full_report += disclaimer
        return full_report

class ReportVisualizer:
    def create_market_share_chart(self, market_share_data, topic_name=""):
        title = f"{topic_name.title()} Market Share Distribution (%)" if topic_name else "Market Share Distribution (%)"
        if not market_share_data or not isinstance(market_share_data, dict) or sum(v for v in market_share_data.values() if isinstance(v, (int,float))) == 0:
            return generate_placeholder_image(text=f"Market Share Data N/A for {topic_name}")
        
        valid_shares = {k: v for k,v in market_share_data.items() if isinstance(v,(int,float)) and v > 0.1} # Min share to show
        if not valid_shares: return generate_placeholder_image(text=f"No Valid Market Share Data for {topic_name}")
        
        sorted_items = sorted(valid_shares.items(), key=lambda x: x[1], reverse=True)
        
        max_slices = 6 # Show up to 5 main + Others
        if len(sorted_items) > max_slices:
            main_players_list = sorted_items[:max_slices-1]
            others_share_total = sum(s_val for _,s_val in sorted_items[max_slices-1:])
            chart_data_list = main_players_list + [("Others", others_share_total)]
        else: 
            chart_data_list = sorted_items
            
        labels = [i[0] for i in chart_data_list]; 
        values = [i[1] for i in chart_data_list]
        return create_pie_chart(title, labels, values)
        
    def create_regional_distribution_chart(self, regions_data, topic_name=""):
        title = f"{topic_name.title()} Regional Market Distribution (%)" if topic_name else "Regional Market Distribution (%)"
        if not regions_data or not isinstance(regions_data, dict) or sum(v for v in regions_data.values() if isinstance(v, (int,float))) == 0:
            return generate_placeholder_image(text=f"Regional Data N/A for {topic_name}")
        labels = list(regions_data.keys()); values = list(regions_data.values())
        return create_bar_chart(title, "Region", "Market Share (%)", labels, values)
        
    def create_segment_distribution_chart(self, segments_data, topic_name=""):
        title = f"{topic_name.title()} Market Segments Distribution (%)" if topic_name else "Market Segments Distribution (%)"
        if not segments_data or not isinstance(segments_data, dict) or sum(v for v in segments_data.values() if isinstance(v, (int,float))) == 0:
            return generate_placeholder_image(text=f"Segment Data N/A for {topic_name}")
        labels = list(segments_data.keys()); values = list(segments_data.values())
        return create_pie_chart(title, labels, values)
        
    def create_forecast_chart(self, forecast_data, topic_name=""):
        title = f"{topic_name.title()} Market Trend Forecast (Illustrative)" if topic_name else "Market Trend Forecast (Illustrative)"
        if not forecast_data or not isinstance(forecast_data, dict) or not any(isinstance(v,list) and len(v)>1 for v in forecast_data.values()):
            return generate_placeholder_image(text=f"Forecast Data N/A for {topic_name}")
        
        valid_fc_data = {}
        for co_name, series_data in forecast_data.items():
            if isinstance(series_data, list) and all(isinstance(x,(int,float)) for x in series_data) and len(series_data)>1:
                valid_fc_data[co_name]=series_data
        if not valid_fc_data: return generate_placeholder_image(text=f"No Valid Forecast Series for {topic_name}")
        
        # Select series to plot (e.g., top 3-4 by final value, plus "Overall" if present)
        plot_data = {}
        if "Overall Market Index" in valid_fc_data:
            plot_data["Overall Market Index"] = valid_fc_data["Overall Market Index"]
        
        # Sort other series by their last value to pick prominent ones
        other_series = {k:v for k,v in valid_fc_data.items() if k != "Overall Market Index"}
        sorted_by_last_val = sorted(other_series.items(), key=lambda item: item[1][-1], reverse=True)
        
        for i in range(min(len(sorted_by_last_val), 4 - len(plot_data))): # Fill up to 4 series total
            plot_data[sorted_by_last_val[i][0]] = sorted_by_last_val[i][1]
            
        if not plot_data: # Fallback if "Overall" wasn't there and nothing else got picked
            plot_data = dict(list(valid_fc_data.items())[:4])

        return create_line_chart(title, "Time Period (e.g., Year)", "Value / Index", plot_data)

    def create_keywords_list_image(self, keywords, topic_name="", width=400, height=350): # Increased height
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        title = f"Key Topics for {topic_name.title()}" if topic_name else "Key Topics / Keywords"
        if not keywords or not isinstance(keywords, list) or not keywords[0]:
            ax.text(0.5, 0.5, "No Keywords Extracted", ha='center', va='center', fontsize=12)
        else:
            ax.set_title(title, fontsize=14, pad=15)
            y_pos = 0.90 # Start a bit lower
            x_start_col1 = 0.05
            x_start_col2 = 0.55
            
            num_keywords_per_col = (len(keywords[:12]) + 1) // 2 # Max 12 keywords in 2 columns

            for i, kw in enumerate(keywords[:12]): # Display up to 12 keywords
                current_x = x_start_col1
                current_y = y_pos - (i % num_keywords_per_col) * 0.09 # Y step based on item in column
                
                if i >= num_keywords_per_col: # Switch to second column
                    current_x = x_start_col2
                
                ax.text(current_x, current_y, f"- {kw.capitalize()}", fontsize=10, va='top')
        
        ax.axis('off')
        plt.tight_layout(pad=1.5) # Add padding
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

class InformationSynthesisSystem: 
    def __init__(self):
        self.researcher = QueryTopicResearcher()
        self.analyzer = DataAnalyzer() 
        self.generator = ReportGenerator()
        self.visualizer = ReportVisualizer()
        
    def process_query(self, query_text):
        # This function now acts as the central orchestrator
        # 1. Get Processed Data (Query Identification, Web Research, Initial Extraction)
        st.session_state.current_step_progress_text = "Interpreting query and fetching initial data..."
        processed_data_payload = self.researcher.get_query_data(query_text)
        
        query_type = processed_data_payload.get("query_type", "general_information")
        report_topic_name = processed_data_payload.get("query_topic", "General Topic").strip()
        if not report_topic_name: report_topic_name = "Undetermined Topic" # Fallback for empty topic
        
        st.session_state.current_topic_name = report_topic_name # For use in other parts like keyword chart title

        sections = []
        visualizations = {}
        report_data_summary_for_ui = {} 

        if query_type == "industry_analysis":
            st.session_state.current_step_progress_text = "Analyzing market data..."
            # 2. In-depth Analysis (for industry reports)
            market_analysis_results = self.analyzer.analyze_market_trends(processed_data_payload)
            competitor_analysis_results = self.analyzer.analyze_competitors(processed_data_payload)
            regional_analysis_results = self.analyzer.analyze_regional_impact(processed_data_payload)
            segment_analysis_results = self.analyzer.analyze_segments(processed_data_payload)
            future_outlook_results = self.analyzer.analyze_future_outlook(processed_data_payload)
            
            st.session_state.current_step_progress_text = "Generating report sections..."
            # 3. Report Section Generation
            sections.append(self.generator.generate_executive_summary(report_topic_name, market_analysis_results, competitor_analysis_results, future_outlook_results))
            sections.append(self.generator.generate_market_overview(report_topic_name, market_analysis_results, regional_analysis_results, segment_analysis_results))
            sections.append(self.generator.generate_competitor_analysis(report_topic_name, competitor_analysis_results))
            sections.append(self.generator.generate_trends_analysis(report_topic_name, market_analysis_results)) # Pass topic name for context
            sections.append(self.generator.generate_strategic_recommendations(report_topic_name, market_analysis_results, competitor_analysis_results, future_outlook_results))
            sections.append(self.generator.generate_future_outlook(report_topic_name, future_outlook_results))
            
            st.session_state.current_step_progress_text = "Creating visualizations..."
            # 4. Visualization Generation
            visualizations["market_share"] = self.visualizer.create_market_share_chart(competitor_analysis_results.get("market_share_data"), report_topic_name)
            visualizations["regional"] = self.visualizer.create_regional_distribution_chart(regional_analysis_results.get("regional_distribution_data"), report_topic_name)
            visualizations["segment"] = self.visualizer.create_segment_distribution_chart(segment_analysis_results.get("segment_distribution_data"), report_topic_name)
            visualizations["forecast"] = self.visualizer.create_forecast_chart(future_outlook_results.get("forecast_chart_data"), report_topic_name)

            # Summary metrics for UI display
            report_data_summary_for_ui = {
                "Market Size": market_analysis_results.get("market_size","N/A"), 
                "CAGR": market_analysis_results.get("growth_rate","N/A"),
                "Identified Players": len(processed_data_payload.get("key_players",[])), 
                "Web Sources Used": len(processed_data_payload.get("fetched_sources",[]))
            }
        elif query_type == "general_information":
            st.session_state.current_step_progress_text = "Extracting summary and keywords..."
            # For general queries, sections are simpler
            sections.append(self.generator.generate_general_introduction(processed_data_payload.get("original_query","N/A"), report_topic_name))
            sections.append(self.generator.generate_general_summary(processed_data_payload.get("summary_sentences")))
            sections.append(self.generator.generate_general_keywords(processed_data_payload.get("keywords")))
            # Optionally add raw text snippet if substantial and desired
            # sections.append(self.generator.generate_text_snippet_section(processed_data_payload.get("raw_text_sample","")))

            st.session_state.current_step_progress_text = "Creating keyword visualization..."
            visualizations["keywords_list"] = self.visualizer.create_keywords_list_image(processed_data_payload.get("keywords", []), report_topic_name)

            report_data_summary_for_ui = {
                "Summary Sentences": len(processed_data_payload.get("summary_sentences",[])),
                "Keywords Found": len(processed_data_payload.get("keywords",[])),
                "Web Sources Used": len(processed_data_payload.get("fetched_sources",[]))
            }

        st.session_state.current_step_progress_text = "Compiling final report..."
        # 5. Compile Full Report (Markdown)
        full_report_md = self.generator.compile_full_report(
            report_topic_name, 
            sections, 
            processed_data_payload.get("fetched_sources"), 
            query_type
        )
        
        return {
            "report_md": full_report_md,
            "topic_name": report_topic_name,
            "query_type": query_type,
            "visualizations": visualizations,
            "ui_summary_metrics": report_data_summary_for_ui 
        }

# --- Documentation Strings ---
APPROACH_EXPLANATION_MD = """
### Approach Explanation

The AI Research & Report Generator adopts a **modular, multi-stage approach** to transform a user's query into a comprehensive report. Our design emphasizes clarity, maintainability, and adaptability.

**Key Principles:**

1.  **Hybrid Data Strategy**: We combine potentially **predefined knowledge** (for well-known industries like Electric Vehicles or AI) with **real-time web scraping** (using DuckDuckGo Search if available, or simulated results otherwise). This allows for both depth in recognized areas and breadth for novel queries.
2.  **Rule-Based NLP & Extraction**: Instead of relying on large, complex NLP models like NLTK's full suite (which was removed), the system employs **custom, lightweight NLP utilities** built with regular expressions (`re` module). This includes sentence tokenization, word tokenization, and a custom stopword list. Data extraction from web pages (market size, CAGR, key players, etc.) also heavily relies on carefully crafted regex patterns. This approach ensures transparency, predictable behavior, and fewer external dependencies.
3.  **Graceful Degradation**: The system is designed to function even if certain components are unavailable. For instance, if the `duckduckgo_search` library isn't installed, it falls back to **simulated search results**, allowing core functionality to remain testable.
4.  **Separation of Concerns**: The system is architected into distinct classes, each handling a specific part of the process:
    *   **Query Interpretation & Orchestration**: Understanding the query's intent and managing data flow.
    *   **Web Research**: Searching the web and fetching/parsing content.
    *   **Data Analysis**: Deriving insights from raw data.
    *   **Report Generation**: Structuring the textual content of the report.
    *   **Visualization**: Creating charts and visual summaries.
5.  **Progressive Report Building**: For industry analyses, the system first tries to populate fields with high-confidence predefined data, then **augments or overrides** it with information extracted from web searches. For unknown industries or general topics, it relies more heavily on web-extracted data and generic templates.
6.  **User-Focused Output**: The final report is presented in a clear Markdown format, supplemented by visualizations to aid comprehension. It includes actionable recommendations (for industry reports) based on the synthesized data.

This approach allows the system to be reasonably robust for a variety of queries, providing useful starting points for research or market understanding, while clearly disclaiming its automated nature and the need for human validation for critical decisions.
"""

SYSTEM_DOCUMENTATION_MD = """
### System Architecture & Workflow

The AI Research & Report Generator is composed of several key Python classes that work in concert:

**Core Components:**

1.  **`InformationSynthesisSystem`**: The main orchestrator. It receives the user query and coordinates the activities of all other components to produce the final report.
2.  **`QueryTopicResearcher`**:
    *   **Query Interpretation**: Uses `identify_query_topic_and_type` to determine if the query is about a known industry (e.g., "electric vehicle market analysis") or a general topic (e.g., "benefits of meditation"). It employs keyword matching and regex for this.
    *   **Data Acquisition Strategy**: Manages the overall data gathering. It leverages `WebResearcher` for online data and merges this with `predefined_industry_data` if the topic is a known industry.
    *   **Data Consolidation**: `get_query_data` is the primary method that returns a structured payload containing all gathered information (text, extracted metrics, sources).
3.  **`WebResearcher`**:
    *   **Web Search**: `search_web_ddg` performs a web search using the DuckDuckGo Search library (if available) or `simulate_search_results` as a fallback. Results are cached.
    *   **Content Fetching**: `get_content_from_url` retrieves HTML content from URLs using the `requests` library, employing user-agent headers and retries. Content is cached.
    *   **Text Extraction**: Uses `BeautifulSoup` to parse HTML and extract meaningful text content, attempting to remove boilerplate like navigation, footers, and scripts.
    *   **Initial Data Extraction**:
        *   `extract_market_data`: For industry-related text, this method uses a series of regular expressions to find and extract specific metrics like market size, CAGR, key players, trends, and challenges.
        *   `extract_general_summary_and_keywords`: For general text, this method uses custom NLP utilities to generate a summary (by scoring sentences based on query relevance and content) and extract key terms.
4.  **`DataAnalyzer`**: (Primarily for industry analysis reports)
    *   Takes the data payload from `QueryTopicResearcher`.
    *   Performs deeper analysis and interpretation:
        *   `analyze_market_trends`: Summarizes market size, growth, and key trends.
        *   `analyze_competitors`: Determines market concentration (e.g., CR4), identifies top players, and describes competitive structure.
        *   `analyze_regional_impact`: Assesses regional distribution and growth.
        *   `analyze_segments`: Looks at market segmentation and dominant/fast-growing segments.
        *   `analyze_future_outlook`: Synthesizes forecast data, opportunities, and challenges into a forward-looking perspective.
5.  **`ReportGenerator`**:
    *   Receives analyzed data from `DataAnalyzer` (for industry reports) or processed data from `QueryTopicResearcher` (for general reports).
    *   Constructs various sections of the report (e.g., Executive Summary, Market Overview, Competitor Analysis, Strategic Recommendations, etc.) in Markdown format. Each section has a dedicated generation method.
    *   `compile_full_report` assembles all sections, adds a title, table of contents, sources appendix, and a disclaimer.
6.  **`ReportVisualizer`**:
    *   Creates visual representations of data using `matplotlib`:
        *   Pie charts for market share and segment distribution (`create_market_share_chart`, `create_segment_distribution_chart`).
        *   Bar charts for regional distribution (`create_regional_distribution_chart`).
        *   Line charts for forecast trends (`create_forecast_chart`).
        *   A simple list-as-image for keywords in general reports (`create_keywords_list_image`).
    *   Outputs images as in-memory byte buffers.
7.  **Custom NLP Utilities**:
    *   `CUSTOM_STOPWORDS`: A predefined set of common English words to be ignored during keyword extraction and text analysis.
    *   `custom_sent_tokenize(text)`: Splits text into sentences using regex (based on punctuation).
    *   `custom_word_tokenize(text)`: Splits text into words and converts to lowercase using regex (alphanumeric sequences).

**Workflow from Input to Output:**

1.  **User Input**: The user enters a query into the Streamlit interface.
2.  **Orchestration Begins (`InformationSynthesisSystem.process_query`)**:
    a.  **Query Processing (`QueryTopicResearcher.get_query_data`)**:
        i.  The query topic and type (industry/general) are identified.
        ii. **Web Research (`WebResearcher.research_topic_online`)**:
            *   A targeted search query is formulated.
            *   Web search is performed.
            *   Content is fetched from top results and parsed.
            *   Initial data (metrics for industry, summary/keywords for general) is extracted from the aggregated web text.
        iii. Data from the web is merged with predefined data (if applicable for known industries).
    b.  **Data Analysis (if Industry Report - `DataAnalyzer` methods)**: The consolidated data is further analyzed to derive deeper insights into market trends, competition, etc.
    c.  **Report Generation (`ReportGenerator` methods)**: Textual sections of the report are generated based on the (analyzed) data.
    d.  **Visualization (`ReportVisualizer` methods)**: Relevant charts and visuals are created.
    e.  **Final Compilation (`ReportGenerator.compile_full_report`)**: All parts are assembled into a single Markdown document.
3.  **Output Display**: The Streamlit UI displays:
    *   The full Markdown report.
    *   Generated visualizations.
    *   Options to download the report and charts.
    *   This documentation.

This structured workflow allows for a systematic approach to information synthesis, from raw query to a structured, insightful report.
"""


# Streamlit UI
def main():
    st.set_page_config(page_title="AI Research & Report Generator", layout="wide")
    
    st.sidebar.title("🧠 AI Report Generator")
    st.sidebar.markdown("Automated insights from web data.")
    
    if not DUCKDUCKGO_SEARCH_AVAILABLE:
        st.sidebar.warning("`duckduckgo_search` lib not found. Web search will be **simulated**. For live results: `pip install duckduckgo-search`", icon="⚠️")

    if 'current_topic_name' not in st.session_state: st.session_state.current_topic_name = "Topic"
    if 'query_input' not in st.session_state: st.session_state.query_input = ""
    if 'current_step_progress_text' not in st.session_state: st.session_state.current_step_progress_text = "Initializing..."


    @st.cache_resource # Cache the system object
    def load_system(): return InformationSynthesisSystem()
    system = load_system()
    
    st.title("AI Research & Report Generator")
    st.subheader("Input a business-level query for industry analysis or any topic for a general summary.")
    
    # Tab for Main App and Documentation
    app_tab, doc_tab = st.tabs(["📄 Report Generation", "ℹ️ System Info & Documentation"])

    with app_tab:
        st.markdown("### Enter Your Query")
        query_examples = [
            "Comprehensive analysis of the electric vehicle market",
            "What are the key trends in the cloud computing industry?",
            "Report on the renewable energy sector: growth, players, and outlook",
            "AI market strategic recommendations",
            "Summarize the impact of climate change on agriculture" # General query example
        ]
        
        st.session_state.query_input = st.text_input(
            "E.g., 'cybersecurity market trends' or 'benefits of renewable energy'", 
            value=st.session_state.query_input, 
            key="query_text_field",
            placeholder="Type your research query here..."
        )
        
        st.markdown("<p style='font-size: smaller;'><b>Example Queries:</b></p>", unsafe_allow_html=True)
        
        # Display example queries in columns for better layout
        num_example_cols = 3 
        example_cols = st.columns(num_example_cols)
        for i, example in enumerate(query_examples):
            if example_cols[i % num_example_cols].button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.query_input = example
                st.rerun() # Rerun to update the input field immediately

        generate_button = st.button("🚀 Generate Report", type="primary", use_container_width=True, disabled=(not st.session_state.query_input.strip()))
        
        if generate_button and st.session_state.query_input.strip():
            query_to_process = st.session_state.query_input
            st.session_state.current_step_progress_text = "Initializing..." # Reset progress text
            
            # Using st.empty for dynamic progress updates
            progress_placeholder = st.empty()

            with st.spinner(f"Researching & generating report for: '{query_to_process[:70]}...' This may take a few moments."):
                start_time = time.time()
                
                # This is a bit of a hack for spinner + progress, normally one would be enough.
                # For complex, long-running tasks, consider background jobs if Streamlit architecture allows.
                # Here, we'll just update a text within the spinner's context.
                # A proper progress bar would require breaking down process_query into yielded steps.
                # For now, we'll use the session state variable to show changing status text.
                
                # Simplified progress indication for this structure:
                progress_bar_ui = st.progress(0, text=st.session_state.current_step_progress_text)
                
                # Simulate progress updates by calling parts of the system and updating bar
                # This is illustrative; true fine-grained progress requires system re-architecture
                
                progress_bar_ui.progress(5, text="Understanding query...")
                # Initial data fetching and query processing part
                # In a real scenario, `system.process_query` would yield progress.
                # Here, we'll call it and then assume phases.
                
                # The `system.process_query` itself will update `st.session_state.current_step_progress_text`
                # We'll update the progress bar based on these assumed stages.
                
                result = system.process_query(query_to_process) # This call is monolithic for progress
                
                # Post-facto progress simulation based on typical stages
                progress_bar_ui.progress(30, text=st.session_state.current_step_progress_text) # After researcher.get_query_data
                time.sleep(0.2) # Simulate work
                progress_bar_ui.progress(60, text=st.session_state.current_step_progress_text) # After analysis
                time.sleep(0.2)
                progress_bar_ui.progress(85, text=st.session_state.current_step_progress_text) # After report gen
                time.sleep(0.2)
                progress_bar_ui.progress(100, text="Report finalized!")
                time.sleep(0.5)
                progress_bar_ui.empty() # Clear the progress bar
            
            end_time = time.time()
            st.success(f"📊 Report for '{result['topic_name'].title()}' ({result['query_type'].replace('_',' ').title()}) generated in {end_time - start_time:.2f} seconds!", icon="✅")
            
            st.markdown("#### Quick Report Metrics:")
            sum_metrics = result["ui_summary_metrics"]
            num_metrics = len(sum_metrics)
            metric_cols = st.columns(num_metrics if num_metrics > 0 else 1)
            
            i = 0
            for key, val in sum_metrics.items():
                metric_cols[i % num_metrics].metric(key, str(val))
                i+=1

            report_display_tab, visuals_tab, export_tab = st.tabs(["📝 Full Report", "📊 Visualizations", "💾 Export Options"])
            
            with report_display_tab:
                st.markdown(result["report_md"], unsafe_allow_html=True) # unsafe_allow_html for download links if embedded
                
            with visuals_tab:
                st.subheader(f"Visual Insights for {result['topic_name'].title()}")
                if result["query_type"] == "industry_analysis":
                    vis_col1, vis_col2 = st.columns(2)
                    with vis_col1:
                        st.image(result["visualizations"]["market_share"], caption="Market Share Distribution", use_column_width=True)
                        st.markdown(get_image_download_link(result["visualizations"]["market_share"], f"{result['topic_name'].replace(' ','_')}_market_share.png"), unsafe_allow_html=True)
                        
                        st.image(result["visualizations"]["segment"], caption="Market Segments", use_column_width=True)
                        st.markdown(get_image_download_link(result["visualizations"]["segment"], f"{result['topic_name'].replace(' ','_')}_segments.png"), unsafe_allow_html=True)
                    with vis_col2:
                        st.image(result["visualizations"]["regional"], caption="Regional Distribution", use_column_width=True)
                        st.markdown(get_image_download_link(result["visualizations"]["regional"], f"{result['topic_name'].replace(' ','_')}_regional.png"), unsafe_allow_html=True)
                        
                        st.image(result["visualizations"]["forecast"], caption="Market Forecast (Illustrative)", use_column_width=True)
                        st.markdown(get_image_download_link(result["visualizations"]["forecast"], f"{result['topic_name'].replace(' ','_')}_forecast.png"), unsafe_allow_html=True)
                elif result["query_type"] == "general_information":
                    st.image(result["visualizations"]["keywords_list"], caption="Extracted Keywords/Topics", use_column_width='auto') # Let Streamlit decide width
                    st.markdown(get_image_download_link(result["visualizations"]["keywords_list"], f"{result['topic_name'].replace(' ','_')}_keywords.png"), unsafe_allow_html=True)
                    st.info("For general topics, primary visualization is keyword-based. More complex NLP could yield network graphs or word clouds.")
                else:
                    st.info("No specific visualizations configured for this report type.")

            with export_tab:
                st.subheader("Export Report")
                report_file_name = f"{result['topic_name'].replace(' ', '_').lower()}_report_{datetime.now().strftime('%Y%m%d')}.md"
                st.download_button(
                    label="Download Full Report as Markdown (.md)",
                    data=result["report_md"],
                    file_name=report_file_name,
                    mime="text/markdown",
                    use_container_width=True
                )
                st.markdown(f"**Report File Name**: `{report_file_name}`")
                st.markdown(f"*Report content generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S %Z')}*")
                st.warning("💡 **Remember**: This AI-generated report synthesizes information from web searches and pre-configured knowledge. Always cross-verify critical data and consult multiple sources for important business decisions.", icon="❗")

        elif generate_button and not st.session_state.query_input.strip():
            st.error("❗ Please enter a query to generate a report.", icon="🚨")
    
    with doc_tab:
        st.header("System Information and Documentation")
        
        st.subheader("1. Approach Explanation")
        st.markdown(APPROACH_EXPLANATION_MD, unsafe_allow_html=True)
        
        st.subheader("2. System Architecture & Workflow")
        st.markdown(SYSTEM_DOCUMENTATION_MD, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"© {datetime.now().year} AI Report Generator\nCustom NLP. Matplotlib Visuals.")

if __name__ == "__main__":
    main()
