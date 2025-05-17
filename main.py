import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import io
import base64
from PIL import Image
import requests
from bs4 import BeautifulSoup
import json
import urllib.parse
from collections import Counter

# Attempt to import duckduckgo_search
try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_SEARCH_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_SEARCH_AVAILABLE = False
    # This warning will be shown in the UI later if needed

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Utility functions (unchanged from original)
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
                    # st.warning(f"Failed to fetch data from {url} after {retries} attempts: {str(e)}") # Too verbose for UI
                    print(f"Warning: Failed to fetch data from {url} after {retries} attempts: {str(e)}")
                    return None
                time.sleep(0.5)
            except Exception as e:
                print(f"Warning: An unexpected error occurred while fetching {url}: {e}")
                return None
        return None
    
    def search_web_ddg(self, query, num_results=3):
        if not DUCKDUCKGO_SEARCH_AVAILABLE:
            # This warning is now handled in main() once at startup
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
                print(f"Warning: DuckDuckGo search for '{query}' yielded no results. Falling back to simulation.")
                return self.simulate_search_results(query, num_results)
            
            self.search_results_cache[cache_key] = results
            return results
        except Exception as e:
            print(f"Error: Error during DuckDuckGo search: {e}. Falling back to simulated results.")
            return self.simulate_search_results(query, num_results)

    def get_content_from_url(self, url):
        if not url: return ""
        if url in self.page_content_cache:
            return self.page_content_cache[url]

        response = self.safe_request(url)
        if response:
            try:
                soup = BeautifulSoup(response.content, 'html.parser')
                for script_or_style in soup(["script", "style", "nav", "footer", "aside"]): # Remove more noise
                    script_or_style.decompose()
                
                text_parts = []
                main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main')
                content_container = main_content if main_content else soup.body # Fallback to body

                if content_container:
                    # Prefer longer paragraphs, then headers, then list items
                    for element in content_container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'div']):
                         # Try to get text from elements that seem to be main content
                        if element.name == 'p' and len(element.get_text(strip=True)) > 50 : # Longer paragraphs
                             text_parts.append(element.get_text(separator=" ", strip=True))
                        elif element.name in ['h1','h2','h3','h4']:
                             text_parts.append(element.get_text(separator=" ", strip=True))
                        elif element.name == 'li' and len(element.get_text(strip=True)) > 10:
                             text_parts.append(element.get_text(separator=" ", strip=True))
                        # For divs, be more selective to avoid grabbing sidebars missed by decompose
                        elif element.name == 'div' and not element.find(['nav','aside','footer']) and len(element.get_text(strip=True)) > 100 :
                            # Check if it has direct text or few children that are not block level
                            if element.find(lambda tag: tag.name not in ['p','h1','h2','h3','h4','li','img','figure','table','ul','ol'] and tag.string and tag.string.strip()):
                                 text_parts.append(element.get_text(separator=" ", strip=True))


                if not text_parts and content_container: # Broader fallback if specific tags yield little
                    text_content_full = content_container.get_text(separator='\n', strip=True)
                    text_parts = [line for line in text_content_full.split('\n') if line and len(line) > 20]


                text_content = "\n".join(text_parts)
                text_content = re.sub(r'\s*\n\s*', '\n', text_content) # Normalize multiple newlines with surrounding spaces
                text_content = re.sub(r'[ \t]+', ' ', text_content)   # Replace multiple spaces/tabs with single space
                text_content = text_content.strip()

                self.page_content_cache[url] = text_content
                return text_content
            except Exception as e:
                print(f"Warning: Error parsing content from {url}: {e}")
        return ""
        
    def research_topic_online(self, topic_query, is_industry_query):
        if is_industry_query:
            search_query_detail = "market analysis report size growth key players trends challenges 2023 2024"
        else:
            search_query_detail = "overview summary key points information"
        search_query = f"{topic_query} {search_query_detail}"

        st.write(f"🌐 Searching web for: '{search_query}'...")
        search_results_metadata = self.search_web_ddg(search_query, num_results=3)

        combined_text_for_processing = ""
        fetched_sources_list = []

        if not search_results_metadata:
            st.warning("No search results found online.")
            market_data_from_web = self.extract_market_data("", topic_query) if is_industry_query else {}
            return market_data_from_web, [], ""

        st.write("📑 Fetching content from search results (top 1-2 for speed):")
        for i, result in enumerate(search_results_metadata[:2]): 
            link = result.get('link')
            title = result.get('title', 'No Title')
            if link:
                st.write(f"   ∟ Fetching: {title} ({link})...")
                content = self.get_content_from_url(link)
                if content:
                    combined_text_for_processing += content + "\n\n" # Add separator
                    fetched_sources_list.append({'title': title, 'link': link})
                    time.sleep(0.1) 
                else:
                    st.write(f"     Failed to fetch content from {link}")
            else:
                st.write(f"  ∟ Skipping result with no link: {title}")

        if not combined_text_for_processing.strip():
             st.warning("Could not fetch detailed content. Using search snippets.")
             combined_text_for_processing = " ".join([res.get('snippet', '') for res in search_results_metadata if res.get('snippet')])
        
        if not combined_text_for_processing.strip():
            st.warning("No text content gathered. Falling back to simulated snippet.")
            sim_results = self.simulate_search_results(topic_query, 1)
            combined_text_for_processing = sim_results[0].get('snippet', '') if sim_results else ""

        market_data_from_web = {}
        if is_industry_query:
            market_data_from_web = self.extract_market_data(combined_text_for_processing, topic_query) 
    
        return market_data_from_web, fetched_sources_list, combined_text_for_processing.strip()

    def extract_market_data(self, combined_text, industry): # For industry queries
        # (This method is largely the same as your previous version, ensure it's robust)
        # ... (Content of extract_market_data from previous version) ...
        market_data = {
            "market_size": "", "cagr": "", "key_players": [], "trends": [], "challenges": [],
            "year": datetime.now().year
        }
        if not combined_text: return market_data

        # Extract market size
        market_size_patterns = [
            r"(?:market size was valued at|valued at|market was worth|current market size is estimated to be|market is estimated at)\s*(USD|EUR|GBP|¥|\$)\s*([\d,]+\.?\d*)\s*(billion|million|trillion)",
            r"(USD|EUR|GBP|¥|\$)\s*([\d,]+\.?\d*)\s*(billion|million|trillion)\s*(?:in market size|in value|as of|in \d{4})",
        ]
        for pattern in market_size_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                market_data["market_size"] = f"{match.group(1)}{match.group(2)} {match.group(3)}"
                break
        if not market_data["market_size"]:
            match = re.search(r"(\$[\d,]+\.?\d*\s*(?:billion|million|trillion))", combined_text, re.IGNORECASE)
            if match: market_data["market_size"] = match.group(1)

        # Extract CAGR
        cagr_patterns = [
            r"CAGR of\s*([\d.]+\s*%)", r"compound annual growth rate.*?\s*([\d.]+\s*%)",
            r"grow at a rate of\s*([\d.]+\s*%)", r"expected to grow at\s*([\d.]+\s*%)",
        ]
        for pattern in cagr_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                market_data["cagr"] = match.group(1).replace(" ", "")
                break
        
        companies = []
        player_intro_patterns = [
            r"(?:key players|major players|leading companies|prominent players|market leaders|significant players)\s*(?:include|are|such as|comprise|operating in this market are)\s*:?\s*([^.]+)\.",
            r"The report profiles key players such as\s*(.*?)(?:and other prominent V\vendors|which include|$)",
        ]
        raw_player_strings = []
        for pattern in player_intro_patterns:
            for match_obj in re.finditer(pattern, combined_text, re.IGNORECASE): raw_player_strings.append(match_obj.group(1))
        
        specific_company_patterns = [
            r"\b(Tesla|BYD|Volkswagen|SAIC|Stellantis|Mercedes-Benz|Ford|General Motors|Hyundai-Kia|Toyota)\b",
            r"\b(Google|Alphabet|Microsoft|Amazon|NVIDIA|IBM|Meta|OpenAI|Anthropic|Baidu|Apple|Salesforce|Oracle)\b",
            r"\b(NextEra Energy|Enel|Iberdrola|EDF|Ørsted|Vestas|First Solar|Canadian Solar|Siemens Gamesa|LONGi)\b",
            r"\b(Amazon Web Services|AWS|Microsoft Azure|Azure|Google Cloud Platform|GCP|Alibaba Cloud|IBM Cloud|Oracle Cloud)\b"
        ]
        for pattern in specific_company_patterns:
            matches = re.findall(pattern, combined_text)
            companies.extend(m for m in matches if m and m not in companies)

        for player_list_str in raw_player_strings:
            potential_players = re.split(r',\s*(?:and\s+)?|\s+and\s+|;\s*', player_list_str)
            for player in potential_players:
                player = player.strip()
                if player and len(player.split()) <= 4 and player[0].isupper() and not player.endswith("etc") and len(player) > 2:
                    if player not in companies: companies.append(player)
        
        seen = set()
        market_data["key_players"] = [x for x in companies if x and not (x in seen or seen.add(x))][:10]

        try:
            sentences = sent_tokenize(combined_text)
            stop_words = set(stopwords.words('english'))

            trend_keywords = ["innovation", "advancement", "growth in", "adoption of", "increasing demand", "shift towards", "emergence of", "rising popularity", "expanding use"]
            trends_found = [s.strip() for s in sentences if any(kw in s.lower() for kw in trend_keywords) and 30 < len(s.strip()) < 250]
            market_data["trends"] = list(dict.fromkeys(trends_found))[:5] # Unique

            challenge_keywords = ["challenge", "obstacle", "barrier", "issue", "concern", "risk", "limitation", "constraint", "difficulty", "threat"]
            challenges_found = [s.strip() for s in sentences if any(kw in s.lower() for kw in challenge_keywords) and 30 < len(s.strip()) < 250]
            market_data["challenges"] = list(dict.fromkeys(challenges_found))[:3] # Unique

        except Exception as e:
            print(f"Warning: NLTK processing error: {e}")
            market_data["trends"] = ["Trend extraction issue."]
            market_data["challenges"] = ["Challenge extraction issue."]
        return market_data

    def extract_general_summary_and_keywords(self, combined_text, query_text, num_summary_sentences=7, num_keywords=10):
        summary_data = {
            "summary_sentences": [], "keywords": [],
            "full_text_snippet": combined_text[:2000] + "..." if len(combined_text) > 2000 else combined_text
        }
        if not combined_text: return summary_data

        try:
            sentences = sent_tokenize(combined_text)
            if not sentences: return summary_data

            stop_words = set(stopwords.words('english'))
            query_words_tokenized = word_tokenize(query_text.lower())
            query_content_words = [w for w in query_words_tokenized if w.isalnum() and w not in stop_words and len(w) > 2]

            # Simple sentence scoring: count query content words
            sentence_scores = []
            for i, s in enumerate(sentences):
                score = 0
                sent_words = word_tokenize(s.lower())
                for qw in query_content_words:
                    if qw in sent_words: score += 1
                # Boost first few sentences slightly as they are often introductory
                if i < 3: score += 0.5 
                if len(s) > 30 and len(s) < 500 : # Filter sentence length
                    sentence_scores.append((s, score, i)) # Keep original index for potential ordering

            # Sort by score (desc) then original index (asc)
            sentence_scores.sort(key=lambda x: (-x[1], x[2]))
            
            # Select top N sentences, trying to maintain some order from original text
            selected_scored_sentences = [s[0] for s in sentence_scores[:num_summary_sentences * 2]] # Get more candidates
            
            # Fallback if scoring yields too few
            if len(selected_scored_sentences) < num_summary_sentences:
                additional_needed = num_summary_sentences - len(selected_scored_sentences)
                # Add from beginning of original text if not already selected
                for s in sentences:
                    if additional_needed == 0: break
                    if s not in selected_scored_sentences:
                        selected_scored_sentences.append(s)
                        additional_needed -=1
            
            summary_data["summary_sentences"] = list(dict.fromkeys(selected_scored_sentences))[:num_summary_sentences]


            words = word_tokenize(combined_text.lower())
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 3]
            
            if filtered_words:
                word_counts = Counter(filtered_words)
                summary_data["keywords"] = [kw for kw, count in word_counts.most_common(num_keywords)]
        except Exception as e:
            print(f"Warning: Error during general text processing: {e}")
            summary_data["summary_sentences"] = [combined_text[:500] + "..."] if combined_text else ["Could not process text."]
        return summary_data

    def simulate_search_results(self, query, num_results=3): # Fallback
        query_lower = query.lower()
        results = []
        generic_results = [
            {'title': f'{query.capitalize()} Overview', 'link': '#', 'snippet': f'This is a general overview of {query_lower}. It covers key aspects and related information.'},
            {'title': f'More about {query.capitalize()}', 'link': '#', 'snippet': f'Further details on {query_lower}, including discussions on its impact and relevance. Market size USD 100 Billion, CAGR 10% if applicable.'},
            {'title': f'{query.capitalize()} Key Points', 'link': '#', 'snippet': f'Important points regarding {query_lower}. Some notable entities are AlphaOrg, BetaCorp.'}
        ]
        # Use specific industry simulation if query matches known ones for better fallback
        if "electric vehicle" in query_lower: results = [{'title': 'EV Overview', 'link': '#', 'snippet': 'Electric vehicles are gaining traction. Tesla, BYD are key players. Market size USD X Billion.'}]
        elif "ai market" in query_lower: results = [{'title': 'AI Summary', 'link': '#', 'snippet': 'Artificial intelligence is a rapidly evolving field. Google, Microsoft lead. CAGR Y%.'}]
        else: results = generic_results
        return results[:num_results]

class QueryTopicResearcher: # Renamed from IndustryResearcher
    def __init__(self):
        self.web_researcher = WebResearcher()
        self.known_industries_keywords_map = { # Map simple key to list of detection keywords
            "electric vehicle": ["electric vehicle", "ev market", "electric car", "bev", "phev"],
            "artificial intelligence": ["artificial intelligence", "ai market", "machine learning", "deep learning", "generative ai"],
            "renewable energy": ["renewable energy", "solar power", "wind energy", "green energy", "clean energy"],
            "cloud computing": ["cloud computing", "iaas", "paas", "saas", "cloud services", "aws", "azure", "gcp"],
        }
        # Predefined data uses the simple keys from above
        self.predefined_industry_data = {
             "electric vehicle": {
                "market_size": "USD 500 Billion (2023)", "cagr": "18.0%",
                "key_players": ["Tesla", "BYD", "Volkswagen", "SAIC", "Stellantis"],
                "trends": ["Rapid adoption of EVs.", "Battery tech advancements.", "Charging infrastructure expansion."],
                "challenges": ["High initial price.", "Range anxiety.", "Supply chain constraints."],
                "market_share": {"Tesla": 20, "BYD": 18, "Volkswagen": 10, "Others": 52},
                "regions": {"Asia-Pacific": 45, "Europe": 30, "North America": 20, "RoW": 5},
                "segments": {"BEV": 70, "PHEV": 30},
                "forecast": { "Tesla": [20,19,18], "BYD": [18,19,20], "Others": [62,62,62]},
                "opportunities": ["Emerging markets.", "Affordable models.", "Autonomous tech integration."]
            },
            "artificial intelligence": {
                "market_size": "USD 200 Billion (2023)", "cagr": "37.0%",
                "key_players": ["Google", "Microsoft", "Amazon", "NVIDIA", "OpenAI"],
                "trends": ["Cross-industry AI/ML adoption.", "Generative AI growth.", "Ethical AI focus."],
                "challenges": ["Data privacy/bias.", "Skilled talent shortage.", "High R&D costs."],
                "market_share": {"Google": 25, "Microsoft": 22, "Amazon": 18, "Others": 35},
                "regions": {"North America": 40, "Asia-Pacific": 30, "Europe": 25, "RoW": 5},
                "segments": {"AI Software": 60, "AI Hardware": 25, "AI Services": 15},
                "forecast": {"Google": [25,24,23], "Microsoft": [22,23,24], "Others": [53,53,53]},
                "opportunities": ["Business process automation.", "Personalized experiences.", "Scientific breakthroughs."]
            },
            # ... add other predefined industries (Renewable Energy, Cloud Computing) here similarly ...
            "renewable energy": {
                "market_size": "USD 1.2 Trillion (2023)", "cagr": "8.5%", "key_players": ["NextEra", "Ørsted", "Enel"],
                "trends": ["Global clean energy transition.","Decreasing solar/wind costs.","Energy storage investment."],
                "challenges": ["Intermittency.","Grid integration.","Land use/env. concerns."],
                "market_share": {"NextEra":15, "Ørsted":12, "Others":73}, "regions": {"Asia-Pacific":40,"Europe":30,"North America":20},
                "segments":{"Solar PV":45,"Wind":35,"Hydro":15},"forecast":{"NextEra":[15,16,17],"Ørsted":[12,13,14],"Others":[73,71,69]},
                "opportunities":["Green hydrogen.","Offshore wind.","Decentralized systems."]
            },
            "cloud computing": {
                "market_size": "USD 700 Billion (2023)", "cagr": "15.0%", "key_players": ["AWS", "Azure", "GCP"],
                "trends": ["Multi/hybrid cloud.","Serverless/containers.","Cloud security focus."],
                "challenges": ["Vendor lock-in.","Multi-cloud management.","Data security/compliance."],
                "market_share": {"AWS":32, "Azure":23, "GCP":11, "Others":34}, "regions": {"North America":50,"Europe":25,"Asia-Pacific":20},
                "segments":{"SaaS":40,"IaaS":35,"PaaS":25},"forecast":{"AWS":[32,31,30],"Azure":[23,24,25],"GCP":[11,12,13],"Others":[34,33,32]},
                "opportunities":["Edge computing.","Cloud AI/ML services.","Industry-specific clouds."]
            }
        }

    def identify_query_topic_and_type(self, query_text):
        query_lower = query_text.lower()
        # Try to match known industry keywords first
        for industry_key, keywords_list in self.known_industries_keywords_map.items():
            for keyword in keywords_list:
                if keyword in query_lower:
                    return industry_key, "industry_analysis" # Return the simple key and type
        
        # If no specific industry keyword match, check for general industry terms
        general_industry_terms = ["market", "industry", "sector", "cagr", "market share", "competitors"]
        if any(term in query_lower for term in general_industry_terms):
            # Try to extract a plausible topic before "market", "industry" etc.
            match = re.search(r"(?:for|of|on|about|analyze|investigate)\s+(?:the\s+)?([\w\s\-]+?)\s+(?:market|industry|sector)", query_lower)
            if match:
                extracted_topic = match.group(1).strip()
                # Check if this extracted topic IS one of our known ones again
                for industry_key, keywords_list in self.known_industries_keywords_map.items():
                    if extracted_topic == industry_key or extracted_topic in keywords_list:
                         return industry_key, "industry_analysis"
                return extracted_topic, "industry_analysis" # Treat as a new industry to research online

        # Default to general information query, use the query itself as topic (or a cleaned version)
        # Clean up the query text a bit to be a "topic"
        topic = re.sub(r"^(?:what is|tell me about|generate a report on|analyze|information on)\s*(?:the\s+)?", "", query_lower, flags=re.IGNORECASE).strip()
        topic = topic.replace("?", "")
        return topic if topic else "general information", "general_information"

    def get_query_data(self, query_text_input):
        identified_topic, query_type = self.identify_query_topic_and_type(query_text_input)
        st.write(f"--- Identified query type: **{query_type}** for topic: **'{identified_topic.title()}'** ---")

        is_industry = (query_type == "industry_analysis")
        # `web_specific_market_data` will be populated if is_industry, otherwise empty dict
        # `combined_web_text` will always be populated with fetched text
        web_specific_market_data, fetched_sources, combined_web_text = self.web_researcher.research_topic_online(identified_topic, is_industry)

        final_data_payload = {}

        if is_industry:
            final_data_payload = self.predefined_industry_data.get(identified_topic, {}).copy() # Start with predefined if available

            if not final_data_payload: # If industry is not predefined, create a default structure
                st.info(f"'{identified_topic.title()}' is treated as an industry for analysis, but has no specific pre-configuration. Report will rely on web search and generic industry templates.")
                final_data_payload = {
                    "market_size": "", "cagr": "", "key_players": [], "trends": [], "challenges": [],
                    "market_share": {}, "regions": {}, "segments": {}, "forecast": {}, "opportunities": [],
                    "year": datetime.now().year
                }
            
            # Merge web-scraped market data into final_data_payload
            if web_specific_market_data.get("market_size"): final_data_payload["market_size"] = web_specific_market_data["market_size"]
            if web_specific_market_data.get("cagr"): final_data_payload["cagr"] = web_specific_market_data["cagr"]
            if web_specific_market_data.get("key_players"): final_data_payload["key_players"] = web_specific_market_data["key_players"]
            if web_specific_market_data.get("trends"): final_data_payload["trends"] = web_specific_market_data["trends"]
            if web_specific_market_data.get("challenges"): final_data_payload["challenges"] = web_specific_market_data["challenges"]

            # Ensure essential fields for industry report have defaults if still empty
            final_data_payload.setdefault("key_players", ["Undetermined Key Players"])
            final_data_payload.setdefault("trends", ["General industry developments observed."])
            final_data_payload.setdefault("challenges", ["Standard competitive and operational challenges."])
            final_data_payload.setdefault("opportunities", ["Exploration of new market segments.", "Technological integration."])
            final_data_payload.setdefault("market_share", {"Top Player (Est.)": 30, "Challenger (Est.)": 20, "Others": 50})
            final_data_payload.setdefault("regions", {"Primary Region": 60, "Secondary Region": 30, "Other Regions": 10})
            final_data_payload.setdefault("segments", {"Main Segment": 70, "Other Segments": 30})
            final_data_payload.setdefault("forecast", {"Overall Market Trend": [100, 110, 121]}) # Index values

        elif query_type == "general_information":
            general_summary_data = self.web_researcher.extract_general_summary_and_keywords(combined_web_text, query_text_input)
            final_data_payload = general_summary_data
            final_data_payload['raw_text_sample'] = combined_web_text[:1500] # For potential display

        final_data_payload["fetched_sources"] = fetched_sources
        final_data_payload["query_type"] = query_type
        final_data_payload["query_topic"] = identified_topic # The determined topic name
        final_data_payload["original_query"] = query_text_input
        
        return final_data_payload

class DataAnalyzer: # Industry-specific analysis
    def analyze_market_trends(self, industry_data):
        # ... (Same as your previous version, ensure it defaults gracefully if data missing)
        market_size = industry_data.get("market_size", "N/A")
        growth_rate = industry_data.get("cagr", "N/A")
        trends = industry_data.get("trends", ["No specific trends identified."])
        
        analysis = {
            "market_size": market_size, "growth_rate": growth_rate,
            "key_trends": trends if trends else ["No specific trends identified."],
            "trend_strength": [random.randint(65, 95) for _ in range(len(trends if trends else [""]))],
            "summary": f"The market is valued at {market_size}, growing at a CAGR of {growth_rate}."
        }
        return analysis

    def analyze_competitors(self, industry_data):
        # ... (Same as your previous version, ensure it defaults gracefully)
        key_players = industry_data.get("key_players", ["Player A", "Player B"])
        market_share = industry_data.get("market_share", {"Player A": 60, "Others": 40})
        
        sorted_shares = sorted([(k, v) for k, v in market_share.items() if k != "Others" and isinstance(v, (int, float)) and v > 0], 
                              key=lambda x: x[1], reverse=True)
        cr4 = sum(s[1] for s in sorted_shares[:4]) if sorted_shares else 0
        
        structure = "Competitive"
        if cr4 > 80: structure = "Highly concentrated"
        elif cr4 > 60: structure = "Concentrated"
        elif cr4 > 40: structure = "Moderately concentrated"
            
        analysis = {
            "key_players": key_players, "market_share": market_share,
            "market_concentration": {"cr4": cr4, "structure": structure},
            "leader_advantage": (sorted_shares[0][1] - sorted_shares[1][1]) if len(sorted_shares) > 1 else (sorted_shares[0][1] if sorted_shares else 0),
            "top_players": [name for name, _ in sorted_shares[:3]] if sorted_shares else ["N/A"]
        }
        return analysis
        
    def analyze_regional_impact(self, industry_data):
        # ... (Same, with defaults)
        regions = industry_data.get("regions", {"Global Focus": 100})
        dominant_region = max(regions.items(), key=lambda x: x[1])[0] if regions else "N/A"
        cagr_val = float(str(industry_data.get("cagr","5%")).replace('%','').replace('N/A','5'))
        growth_rates = {r: round(random.uniform(max(1,cagr_val*0.5), cagr_val*1.5),1) for r in regions.keys()}
        fastest_growing = max(growth_rates.items(), key=lambda x: x[1])[0] if growth_rates else "N/A"
        analysis = {
            "regional_distribution": regions, "dominant_region": dominant_region,
            "regional_growth": growth_rates, "fastest_growing": fastest_growing,
            "emerging_markets": [r for r,s in regions.items() if s < 15 and growth_rates.get(r,0) > (cagr_val*0.8)]
        }
        return analysis
        
    def analyze_segments(self, industry_data):
        # ... (Same, with defaults)
        segments = industry_data.get("segments", {"Primary Segment": 100})
        dominant_segment = max(segments.items(), key=lambda x: x[1])[0] if segments else "N/A"
        cagr_val = float(str(industry_data.get("cagr","5%")).replace('%','').replace('N/A','5'))
        growth_rates = {s: round(random.uniform(max(1,cagr_val*0.6), cagr_val*1.8),1) for s in segments.keys()}
        fastest_growing = max(growth_rates.items(), key=lambda x: x[1])[0] if growth_rates else "N/A"
        analysis = {
            "segment_distribution": segments, "dominant_segment": dominant_segment,
            "segment_growth": growth_rates, "fastest_growing": fastest_growing
        }
        return analysis
        
    def analyze_future_outlook(self, industry_data):
        # ... (Same, with defaults)
        forecast = industry_data.get("forecast", {"Overall Market": [10,12,15]})
        if not isinstance(forecast, dict) or not forecast: forecast = {"Overall Market": [10,12,15]}
        for k,v in forecast.items(): # Ensure values are numeric lists
            if not isinstance(v, list) or not all(isinstance(x,(int,float)) for x in v):
                forecast[k] = [random.randint(10,20) for _ in range(3)]

        challenges = industry_data.get("challenges", ["Generic challenge 1"])
        opportunities = industry_data.get("opportunities", ["Generic opportunity 1"])
        trend_analysis = {}
        for co, data in forecast.items():
            if len(data)>1:
                trend = data[-1]-data[0]; pc = round((data[-1]-data[0])/data[0]*100,1) if data[0]!=0 else 0
                trend_analysis[co]={"data":data,"overall_trend":trend,"direction":"increasing" if trend>0 else "decreasing","percent_change":pc}
        growing_cos = sorted([(k,v["percent_change"]) for k,v in trend_analysis.items() if v["percent_change"]>0],key=lambda x:x[1],reverse=True)
        analysis = {
            "forecast_data": forecast, "trend_analysis": trend_analysis,
            "fastest_growing_players": [n for n,_ in growing_cos[:3]] if growing_cos else ["N/A"],
            "key_challenges": challenges, "key_opportunities": opportunities
        }
        return analysis

class ReportGenerator:
    # --- Industry Specific Sections ---
    def generate_executive_summary(self, topic_name, market_analysis, competitor_analysis):
        # ... (largely same as your previous good version, ensure graceful N/A handling)
        market_size = market_analysis.get("market_size", "N/A")
        growth_rate = market_analysis.get("growth_rate", "N/A")
        top_players_list = competitor_analysis.get("top_players", ["Key industry participants"])
        top_players = ", ".join(top_players_list[:3]) if top_players_list and top_players_list[0] != "N/A" else "Key industry participants"
        
        structure = competitor_analysis.get("market_concentration", {}).get("structure", "competitive")
        cr4 = competitor_analysis.get("market_concentration", {}).get("cr4", 0)

        trends = market_analysis.get("key_trends", ["general industry developments", "further market evolution"])
        trend1 = trends[0].lower().strip('.') if trends else "general industry developments"
        trend2 = trends[1].lower().strip('.') if len(trends) > 1 else "further market evolution"

        return f"""## Executive Summary\nThe {topic_name.title()} industry, valued at {market_size}, is projected to grow at a CAGR of {growth_rate}. Key findings:
1. **Market Structure**: A {structure} market, with top four players holding ~{cr4:.1f}% share.
2. **Leading Companies**: {top_players} are dominant.
3. **Growth Opportunities**: Significant potential in areas like {trend1} and {trend2}.
This report offers strategic insights for success."""

    def generate_market_overview(self, topic_name, market_analysis, regional_analysis, segment_analysis):
        # ... (largely same as your previous good version, ensure graceful N/A handling)
        ma = market_analysis; ra = regional_analysis; sa = segment_analysis
        trends = ma.get("key_trends", ["a key development", "another shift"])
        return f"""## Market Overview
### Market Size and Growth
The {topic_name.title()} market at {ma.get("market_size","N/A")} is expected to grow at {ma.get("growth_rate","N/A")} CAGR.
### Regional Distribution
{ra.get("dominant_region","N/A")} leads with {ra.get("regional_distribution",{}).get(ra.get("dominant_region","N/A"),"N/A")}%. Fastest growing: {ra.get("fastest_growing","N/A")} ({ra.get("regional_growth",{}).get(ra.get("fastest_growing","N/A"),"N/A")}%).
### Market Segments
Segments: {", ".join(list(sa.get("segment_distribution",{}).keys()))}. Dominant: {sa.get("dominant_segment","N/A")} ({sa.get("segment_distribution",{}).get(sa.get("dominant_segment","N/A"),"N/A")}%). Fastest: {sa.get("fastest_growing","N/A")} ({sa.get("segment_growth",{}).get(sa.get("fastest_growing","N/A"),"N/A")}%).
### Key Market Drivers
1. {trends[0].strip('.') if trends else 'N/A'}.
2. {trends[1].strip('.') if len(trends)>1 else 'N/A'}."""

    def generate_competitor_analysis(self, topic_name, competitor_analysis):
        # ... (largely same as your previous good version, ensure graceful N/A handling)
        ca = competitor_analysis; mc = ca.get("market_concentration",{})
        sorted_comp = sorted([(k,v) for k,v in ca.get("market_share",{}).items() if k!="Others" and v>0], key=lambda x:x[1], reverse=True)[:5]
        players_str = ""
        if not sorted_comp: players_str = "Detailed competitor data is limited."
        else:
            for i, (co, sh) in enumerate(sorted_comp):
                pos = ["leading","second","third","fourth","fifth"][i] if i<5 else "significant"
                players_str += f"**{co}** ({pos}, ~{sh:.1f}% share). "
                if i==0: players_str+="Sets benchmarks.\n"
                elif i==1: players_str+="Challenges leadership.\n"
                else: players_str+="Contributes to diversity.\n"

        return f"""## Competitor Analysis
### Market Concentration
{topic_name.title()} market is {mc.get("structure","competitive")}. Top 4: {mc.get("cr4",0):.1f}%.
### Key Players
{players_str}
### Competitive Dynamics
Landscape: {"intense" if ca.get("leader_advantage",0)<8 else "stable"}. Leader vs challenger gap: {ca.get("leader_advantage",0):.1f} pts."""

    def generate_trends_analysis(self, market_analysis):
        # ... (largely same as your previous good version)
        trends = market_analysis.get("key_trends", [])
        strengths = market_analysis.get("trend_strength", [])
        analysis = "## Key Industry Trends\nInfluential trends shaping the industry:\n"
        if not trends: analysis += "Specific trends are under observation. General trends include tech advancements.\n"
        else:
            for i, trend in enumerate(trends[:3]): # Max 3 for brevity
                st_val = strengths[i] if i < len(strengths) else random.randint(70,90)
                imp = "transformative" if st_val > 85 else "significant"
                analysis += f"\n### {i+1}. {trend.strip('.')}\n**Impact**: {imp.title()} (Est. {st_val}/100). Adaptation is key.\n"
        return analysis

    def generate_strategic_recommendations(self, topic_name, market_analysis, competitor_analysis, future_analysis):
        # ... (largely same as your previous good version)
        fa = future_analysis; ca = competitor_analysis; ma = market_analysis
        ops = fa.get("key_opportunities",["new tech","customer engagement"])[:2]
        chs = fa.get("key_challenges",["econ uncertainty","talent shortage"])[:2]
        mc_cr4 = ca.get("market_concentration",{}).get("cr4",50)
        
        return f"""## Strategic Recommendations
For {topic_name.title()} market:
1. **Market Navigation**: {'Focus on niches if CR4 > 70%' if mc_cr4 > 70 else 'Innovate to gain share if CR4 < 70%'}.
2. **Investment**: Prioritize {ops[0].lower()} & {ops[1].lower()}. Mitigate {chs[0].lower()}.
3. **Competitive Strategy**: Analyze fast-growing players. Agility is crucial.
4. **Risk Management**: Address {chs[1].lower()}.
5. **Innovation**: {'Accelerate tech adoption if key driver' if 'tech' in str(ma.get('key_trends',[])).lower() else 'Use tech for efficiency'}.
6. **Ecosystem**: Explore alliances to address challenges or access new markets."""

    def generate_future_outlook(self, topic_name, future_analysis):
        # ... (largely same as your previous good version)
        fa = future_analysis
        gps = fa.get("fastest_growing_players",["Emerging players"])
        kcs = fa.get("key_challenges",["Evolving demands","New regulations"])[:2]
        kos = fa.get("key_opportunities",["Untapped segments","Innovative products"])[:2]
        return f"""## Future Market Outlook
### Market Evolution (3-5 Years)
{topic_name.title()} market will see:
1. **Dynamics**: {'Consolidation or new niches' if len(gps)<3 else 'Dynamic with multiple players'}.
2. **Shifts**: {gps[0] if gps else 'Innovators'} may disrupt leaders.
3. **Challenges**: {kcs[0].strip('.')} will be prominent.
### Critical Uncertainties
1. **Regulatory**: Impact from changes related to {kcs[1].lower().strip('.')}.
2. **Technology**: Pace of breakthroughs in {kos[0].lower().strip('.')}.
### Long-Term Opportunities
1. **{kos[0].strip('.')}**: Substantial growth potential.
2. **{kos[1].strip('.')}**: Early engagement could lead to market leadership."""

    # --- General Information Sections ---
    def generate_general_introduction(self, original_query, topic_name):
        return f"""## Introduction
This report provides a summary of information related to your query: "{original_query}". The research focuses on the topic of **{topic_name.title()}** based on publicly available web content."""

    def generate_general_summary(self, summary_sentences):
        if not summary_sentences or not isinstance(summary_sentences, list) or not summary_sentences[0]:
            return "## Key Information Summary\nNo specific summary points could be extracted from the web content.\n"
        
        summary_points = "\n".join([f"- {s.strip()}" for s in summary_sentences])
        return f"""## Key Information Summary
Based on the web research, the following key points or summary sentences were extracted:
{summary_points}"""

    def generate_general_keywords(self, keywords):
        if not keywords or not isinstance(keywords, list) or not keywords[0]:
            return "\n## Main Keywords/Topics\nNo distinct keywords were identified from the content.\n"
            
        keywords_list = ", ".join([kw.capitalize() for kw in keywords])
        return f"""## Main Keywords/Topics
The research identified the following as prominent keywords or topics within the fetched content:
- {keywords_list}"""
    
    def generate_text_snippet_section(self, text_snippet):
        if not text_snippet: return ""
        return f"""\n## Extended Content Snippet
Below is a more extensive snippet from the aggregated web content:
\n---\n{text_snippet}\n---"""

    def compile_full_report(self, topic_name, sections, fetched_sources=None, query_type="industry_analysis"):
        current_date = datetime.now().strftime("%B %d, %Y")
        
        report_title_main = f"{topic_name.title()} Information Report"
        report_subtitle = "Key Insights and Summary from Web Research"
        toc_items = ["Introduction", "Key Information Summary", "Main Keywords/Topics"] # Default for general

        if query_type == "industry_analysis":
            report_title_main = f"{topic_name.title()} Industry Intelligence Report"
            report_subtitle = "Comprehensive Market Analysis and Strategic Recommendations"
            toc_items = ["Executive Summary","Market Overview","Competitor Analysis","Key Industry Trends","Strategic Recommendations","Future Market Outlook"]

        title_md = f"# {report_title_main}\n### {report_subtitle}\n#### {current_date}\n"
        
        toc_md = "\n## Table of Contents\n"
        for i, item_name in enumerate(toc_items):
            anchor = item_name.lower().replace(" ", "-").replace("/", "") # Basic anchor
            toc_md += f"{i+1}. [{item_name}](#{anchor})\n"

        appendix_toc_num = len(toc_items) + 1
        if fetched_sources:
            toc_md += f"{appendix_toc_num}. [Data Sources Appendix](#data-sources-appendix)\n"

        full_report = title_md + toc_md + "".join(sections)
        
        if fetched_sources:
            sources_appendix = f"\n## Data Sources Appendix\nThis report utilized information from:\n"
            for i, source in enumerate(fetched_sources):
                sources_appendix += f"- [{source.get('title','Source')}]({source.get('link','#')})\n"
            full_report += sources_appendix
        
        disclaimer = """\n---\n*Disclaimer: This AI-generated report is based on web searches and predefined data (if applicable). Use as one of many inputs for decision-making. Information reflects content at time of generation.*"""
        full_report += disclaimer
        return full_report

class ReportVisualizer:
    def create_market_share_chart(self, market_share_data):
        if not market_share_data or not isinstance(market_share_data, dict) or sum(v for v in market_share_data.values() if isinstance(v, (int,float))) == 0:
            return generate_placeholder_image(text="Market Share Data N/A")
        valid_shares = {k: v for k,v in market_share_data.items() if isinstance(v,(int,float)) and v > 0}
        if not valid_shares: return generate_placeholder_image(text="No Valid Market Share Data")
        # ... (rest of pie chart logic from before)
        sorted_items = sorted(valid_shares.items(), key=lambda x: x[1], reverse=True)
        max_slices = 5
        if len(sorted_items) > max_slices:
            main_players = sorted_items[:max_slices-1]; others_share = sum(s for _,s in sorted_items[max_slices-1:])
            chart_data_list = main_players + [("Others", others_share)]
        else: chart_data_list = sorted_items
        labels = [i[0] for i in chart_data_list]; values = [i[1] for i in chart_data_list]
        return create_pie_chart("Market Share Distribution (%)", labels, values)
        
    def create_regional_distribution_chart(self, regions_data):
        if not regions_data or not isinstance(regions_data, dict) or sum(v for v in regions_data.values() if isinstance(v, (int,float))) == 0:
            return generate_placeholder_image(text="Regional Data N/A")
        labels = list(regions_data.keys()); values = list(regions_data.values())
        return create_bar_chart("Regional Market Distribution (%)", "Region", "Market Share (%)", labels, values)
        
    def create_segment_distribution_chart(self, segments_data):
        if not segments_data or not isinstance(segments_data, dict) or sum(v for v in segments_data.values() if isinstance(v, (int,float))) == 0:
            return generate_placeholder_image(text="Segment Data N/A")
        labels = list(segments_data.keys()); values = list(segments_data.values())
        return create_pie_chart("Market Segments Distribution (%)", labels, values)
        
    def create_forecast_chart(self, forecast_data):
        if not forecast_data or not isinstance(forecast_data, dict) or not any(v for v in forecast_data.values() if isinstance(v,list)):
            return generate_placeholder_image(text="Forecast Data N/A")
        # ... (rest of line chart logic from before, ensuring robustness)
        valid_fc_data = {}
        for co, series in forecast_data.items():
            if isinstance(series, list) and all(isinstance(x,(int,float)) for x in series) and len(series)>1:
                valid_fc_data[co]=series
        if not valid_fc_data: return generate_placeholder_image(text="No Valid Forecast Series")
        
        if len(valid_fc_data)>4: # Limit lines
            sorted_by_last = sorted(valid_fc_data.items(), key=lambda x:x[1][-1], reverse=True)
            plot_data = dict(sorted_by_last[:3])
            if "Others" in valid_fc_data: plot_data["Others"]=valid_fc_data["Others"]
        else: plot_data = valid_fc_data
        return create_line_chart("Market Trend Forecast (Illustrative)", "Time Period", "Value/Index", plot_data)

    def create_keywords_list_image(self, keywords, width=400, height=300):
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        if not keywords or not isinstance(keywords, list) or not keywords[0]:
            ax.text(0.5, 0.5, "No Keywords Extracted", ha='center', va='center', fontsize=12)
        else:
            ax.set_title("Key Topics / Keywords", fontsize=14)
            y_pos = 0.95; x_start = 0.05
            for i, kw in enumerate(keywords[:10]): # Display top 10
                ax.text(x_start, y_pos, f"- {kw.capitalize()}", fontsize=10, va='top')
                y_pos -= 0.085
                if i == 4: # Start a new column for >5 keywords
                    y_pos = 0.95; x_start = 0.55
                if y_pos < 0.1 and x_start == 0.55: break # Max 2 columns
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

class InformationSynthesisSystem: # Renamed from IndustryIntelligenceSystem
    def __init__(self):
        self.researcher = QueryTopicResearcher()
        self.analyzer = DataAnalyzer() # Industry-specific analyzer
        self.generator = ReportGenerator()
        self.visualizer = ReportVisualizer()
        
    def process_query(self, query_text):
        processed_data = self.researcher.get_query_data(query_text)
        
        query_type = processed_data.get("query_type", "general_information")
        report_topic_name = processed_data.get("query_topic", "General Topic")
        st.session_state.current_topic_name = report_topic_name 

        sections = []
        visualizations = {}
        report_data_summary = {} # For UI display of what was found

        if query_type == "industry_analysis":
            market_analysis = self.analyzer.analyze_market_trends(processed_data)
            competitor_analysis = self.analyzer.analyze_competitors(processed_data)
            regional_analysis = self.analyzer.analyze_regional_impact(processed_data)
            segment_analysis = self.analyzer.analyze_segments(processed_data)
            future_analysis = self.analyzer.analyze_future_outlook(processed_data)
            
            sections.append(self.generator.generate_executive_summary(report_topic_name, market_analysis, competitor_analysis))
            sections.append(self.generator.generate_market_overview(report_topic_name, market_analysis, regional_analysis, segment_analysis))
            sections.append(self.generator.generate_competitor_analysis(report_topic_name, competitor_analysis))
            sections.append(self.generator.generate_trends_analysis(market_analysis))
            sections.append(self.generator.generate_strategic_recommendations(report_topic_name, market_analysis, competitor_analysis, future_analysis))
            sections.append(self.generator.generate_future_outlook(report_topic_name, future_analysis))
            
            visualizations["market_share"] = self.visualizer.create_market_share_chart(processed_data.get("market_share"))
            visualizations["regional"] = self.visualizer.create_regional_distribution_chart(processed_data.get("regions"))
            visualizations["segment"] = self.visualizer.create_segment_distribution_chart(processed_data.get("segments"))
            visualizations["forecast"] = self.visualizer.create_forecast_chart(processed_data.get("forecast"))

            report_data_summary = {
                "Market Size": processed_data.get("market_size","N/A"), "CAGR": processed_data.get("cagr","N/A"),
                "Key Players": len(processed_data.get("key_players",[])), "Sources Used": len(processed_data.get("fetched_sources",[]))
            }
        elif query_type == "general_information":
            sections.append(self.generator.generate_general_introduction(processed_data.get("original_query","N/A"), report_topic_name))
            sections.append(self.generator.generate_general_summary(processed_data.get("summary_sentences")))
            sections.append(self.generator.generate_general_keywords(processed_data.get("keywords")))
            # Optionally add a snippet of the raw text if useful
            # sections.append(self.generator.generate_text_snippet_section(processed_data.get("raw_text_sample","")))


            visualizations["keywords_list"] = self.visualizer.create_keywords_list_image(processed_data.get("keywords", []))
            # Could add a word cloud here if library is added

            report_data_summary = {
                "Summary Sentences": len(processed_data.get("summary_sentences",[])),
                "Keywords Found": len(processed_data.get("keywords",[])),
                "Sources Used": len(processed_data.get("fetched_sources",[]))
            }

        full_report_md = self.generator.compile_full_report(report_topic_name, sections, processed_data.get("fetched_sources"), query_type)
        
        return {
            "report_md": full_report_md,
            "topic_name": report_topic_name,
            "query_type": query_type,
            "visualizations": visualizations,
            "ui_summary_metrics": report_data_summary 
        }

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Research & Report Generator", layout="wide")
    
    st.title("🧠 AI Research & Report Generator")
    st.subheader("Get insights on any topic or industry analysis using web data.")
    
    if not DUCKDUCKGO_SEARCH_AVAILABLE:
        st.warning("`duckduckgo_search` library not found. Web search will be simulated. For live web search, please install it: `pip install duckduckgo-search`", icon="⚠️")


    if 'current_topic_name' not in st.session_state: st.session_state.current_topic_name = ""
    if 'query_input' not in st.session_state: st.session_state.query_input = ""

    @st.cache_resource
    def load_system(): return InformationSynthesisSystem()
    system = load_system()
    
    st.markdown("### Enter Your Query")
    query_examples = [
        "Generate a strategy report for the electric vehicle market",
        "What are the benefits of meditation?",
        "Analyze the cloud computing industry",
        "History of Python programming language",
        "Renewable energy market trends and opportunities"
    ]
    
    st.session_state.query_input = st.text_input("Enter your query (e.g., 'AI market trends' or 'effects of climate change')", 
                                                 value=st.session_state.query_input, key="query_text_field")
    
    st.markdown("**Example queries:**")
    cols = st.columns(len(query_examples))
    for i, example in enumerate(query_examples):
        if cols[i].button(example, key=f"example_{i}", use_container_width=True):
            st.session_state.query_input = example
            st.experimental_rerun()

    generate_button = st.button("🚀 Generate Report", type="primary", use_container_width=True, disabled=(not st.session_state.query_input))
    
    if generate_button and st.session_state.query_input:
        query_to_process = st.session_state.query_input
        with st.spinner(f"Researching and generating report for: '{query_to_process[:60]}...' This may take moments."):
            progress_bar = st.progress(0, text="Initializing...")
            start_time = time.time()
            
            # Update progress helper
            def update_progress(value, text):
                progress_bar.progress(value, text=text)
                # time.sleep(0.05) # Minimal sleep for UX

            update_progress(5, "Understanding query...")
            result = system.process_query(query_to_process) # This now includes web search & type detection
            # Progress updates happen via st.write within the system for now
            
            update_progress(70, "Analyzing information...")
            update_progress(85, "Compiling report sections...")
            update_progress(95, "Creating visualizations...")
            
            progress_bar.progress(100, text="Report finalized!")
            time.sleep(0.3)
            progress_bar.empty()
        
        end_time = time.time()
        st.success(f"📊 Report for '{result['topic_name'].title()}' ({result['query_type']}) generated in {end_time - start_time:.2f} seconds!")
        
        st.markdown("#### Summary of Fetched Data:")
        sum_metrics = result["ui_summary_metrics"]
        num_metrics = len(sum_metrics)
        metric_cols = st.columns(num_metrics if num_metrics > 0 else 1)
        
        i = 0
        for key, val in sum_metrics.items():
            metric_cols[i % num_metrics].metric(key, str(val))
            i+=1

        report_tab, visuals_tab, export_tab = st.tabs(["📝 Full Report", "📊 Visualizations", "💾 Export Options"])
        
        with report_tab:
            st.markdown(result["report_md"])
            
        with visuals_tab:
            st.subheader(f"Visual Insights for {result['topic_name'].title()}")
            if result["query_type"] == "industry_analysis":
                vis_col1, vis_col2 = st.columns(2)
                with vis_col1:
                    st.image(result["visualizations"]["market_share"], caption="Market Share Distribution", use_column_width=True)
                    st.markdown(get_image_download_link(result["visualizations"]["market_share"], "market_share.png"), unsafe_allow_html=True)
                    st.image(result["visualizations"]["segment"], caption="Market Segments", use_column_width=True)
                    st.markdown(get_image_download_link(result["visualizations"]["segment"], "segments.png"), unsafe_allow_html=True)
                with vis_col2:
                    st.image(result["visualizations"]["regional"], caption="Regional Distribution", use_column_width=True)
                    st.markdown(get_image_download_link(result["visualizations"]["regional"], "regional.png"), unsafe_allow_html=True)
                    st.image(result["visualizations"]["forecast"], caption="Market Forecast (Illustrative)", use_column_width=True)
                    st.markdown(get_image_download_link(result["visualizations"]["forecast"], "forecast.png"), unsafe_allow_html=True)
            elif result["query_type"] == "general_information":
                st.image(result["visualizations"]["keywords_list"], caption="Extracted Keywords/Topics", use_column_width=False)
                st.markdown(get_image_download_link(result["visualizations"]["keywords_list"], "keywords.png"), unsafe_allow_html=True)
                st.markdown("Further visualizations for general topics (e.g., word clouds) could be added here.")
            else:
                st.info("No specific visualizations for this report type yet.")

        with export_tab:
            st.subheader("Export Report")
            st.download_button(
                label="Download Report as Markdown (.md)",
                data=result["report_md"],
                file_name=f"{result['topic_name'].replace(' ', '_').lower()}_report.md",
                mime="text/markdown",
                use_container_width=True
            )
            st.markdown(f"*Report generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}*")
            st.info("💡 This AI-generated report synthesizes information from web searches and pre-configured knowledge. Always cross-verify critical data for important decisions.")

    elif generate_button and not st.session_state.query_input:
        st.warning("Please enter a query to generate a report.")

    st.markdown("---")
    st.caption("AI Research & Report Generator. Web search via DuckDuckGo (if lib available).")

if __name__ == "__main__":
    main()