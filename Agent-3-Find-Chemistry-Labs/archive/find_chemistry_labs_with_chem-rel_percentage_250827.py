import os
import json

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from google import genai
from google.genai import types
from google.genai.types import Tool, GoogleSearch

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException

from bs4 import BeautifulSoup
import argparse
import re
from collections import Counter

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run in background
chrome_options.add_argument('--window-size=1920,1080')

# Create driver
driver = webdriver.Chrome(options=chrome_options)
driver.set_page_load_timeout(15)  # 15 second timeout


load_dotenv()

def load_existing_results(output_file, id_key='id', required_key=None):
    """Load existing results to enable interrupted processing continuation."""
    processed_ids = set()
    existing_results = []
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                for entry in existing_results:
                    if id_key in entry and (required_key is None or required_key in entry):
                        processed_ids.add(entry[id_key])
                print(f"Found {len(processed_ids)} already processed labs in {output_file}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not read existing results from {output_file}: {e}")
            print("Starting fresh...")
    
    return processed_ids, existing_results

def initialize_output_file(output_file, df_length):
    """Initialize output JSON file with empty array if needed."""
    if df_length > 0 and not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)

def prepare_lab_info(lab, exclude_columns=None):
    """Prepare lab information string from lab data."""
    if exclude_columns is None:
        exclude_columns = ['id', 'logo url', 'standards', 'qualifications', 'gallery', 'publications']
    
    lab_info = ""
    for k, v in lab.to_dict().items():
        if (pd.notna(v) and 
            str(v).strip().lower() not in ['', 'nan', 'none'] and 
            k not in exclude_columns):
            lab_info += f"{k}: {v}, "
    return lab_info

def create_gemini_client(model_version="flash", use_search=False):
    """Create and configure Gemini client with tools."""
    client = genai.Client()
    config = {
        "response_modalities": ["TEXT"],
        "thinking_config": types.ThinkingConfig(thinking_budget=-1)
    }
    
    if use_search:
        google_search_tool = Tool(google_search=GoogleSearch())
        config["tools"] = [google_search_tool]
    
    model_name = f"gemini-2.5-{model_version}"
    return client, model_name, config

def parse_numbered_response(response_text, expected_items=6):
    """Parse numbered list response format from AI models."""
    if not response_text:
        return [''] * expected_items
    
    lines = response_text.split('\n')
    parsed_items = [''] * expected_items
    
    for i in range(1, expected_items + 1):
        line = next((line for line in lines if line.strip().startswith(f'{i}.')), '')
        if line:
            parsed_items[i-1] = line.replace(f'{i}.', '').strip()
    
    return parsed_items

def save_results_incrementally(results, output_file, current_index, total_count):
    """Save results incrementally with progress information."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved lab {current_index + 1}/{total_count} to {output_file}")

def load_labs_json(filename):
    """Load and parse the input JSON file containing lab information."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dictionary formats
        if isinstance(data, list):
            # If data is a list, convert directly to DataFrame
            df = pd.DataFrame(data)
            print(f"Loaded {len(df)} labs from {filename} (list format)")
        elif isinstance(data, dict):
            # If data is a dictionary, convert with index
            df = pd.DataFrame.from_dict(data, orient='index')
            print(f"Loaded {len(df)} labs from {filename} (dict format)")
        else:
            print(f"Error: Unsupported JSON format in {filename}")
            return None
        
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def fetch_url(url: str) -> dict:
    """Fetch URL using ChromeDriver and return response with status code and content."""
    try:
        try:
            # Navigate to URL
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            # Get page content
            page_source = driver.page_source
            current_url = driver.current_url
            title = driver.title
            
            # Parse HTML to extract text content
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Get text content
            text_content = soup.get_text()
            
            # Clean up text - remove extra whitespace and newlines
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Get status code (approximate - Chrome doesn't expose this directly)
            status_code = 200  # Assume success if we got here
            
            return {
                'status_code': status_code,
                'content': clean_text,
                'title': title,
                'url': current_url,  # Final URL after redirects
                'headers': {}  # ChromeDriver doesn't expose headers easily
            }
            
        finally:
            driver.close()
            
    except TimeoutException:
        return {
            'status_code': 0,
            'content': 'Error: Page load timeout',
            'title': '',
            'url': url,
            'headers': {}
        }
    except WebDriverException as e:
        return {
            'status_code': 0,
            'content': f'WebDriver error: {str(e)}',
            'title': '',
            'url': url,
            'headers': {}
        }
    except Exception as e:
        return {
            'status_code': 0,
            'content': f'Error fetching URL: {str(e)}',
            'title': '',
            'url': url,
            'headers': {}
        }

def research_lab(lab_info: str, model_version="pro"):
    """Research lab to determine if it offers analytical chemistry services."""
    system_prompt = """
        You are a research assistant helping to identify analytical chemistry labs.
        """

    user_prompt = """
    Please research this laboratory to determine if it offers analytical chemistry services or any chemistry-related testing. 
    
    The following services are generally not considered analytical chemistry services/testing:
    - Biological testing
    - clinical testing
    - microbiology testing
    - healthcare testing
    - Mechanical testing
    - Electrical testing
    
    Examples of analytical chemistry services (include but not limit to):
    - Chemical analysis
    - Analytical chemistry
    - Chemical testing
    - Material composition analysis
    - Environmental chemistry testing
    - Food chemistry analysis
    - Pharmaceutical analysis
    
    ## Information source
    1. The lab info provided below such as lab name, address, or description.
    2. Web search results about this lab.
    3. Some lab's info may include links to the lab's website and social media profiles. If so, please be sure to visit those links and include the information there in your analysis. 

    ## Hints on composing queries
    1. You should not rely on a single, initial query.
    2. Try to come up with different queries by combining the lab name with different type of info of the lab, such as "lab name + testing type", "lab name + overview", "lab name + address". 
    3. Iteratively search, analyze, and repeat. At each round, analyze the search results to compose new queries for the next round of search.

    ## How to reach a conclusion
    * Only use the information source above to reach a conclusion. Do not make assumptions or use your own knowledge to make a judgement.
    * You must find quotes from the information source, including lab info given below, web search results you retrieved using the Google Search tool, and the lab's website and social media profiles (if available) you visited, to reach the conclusion. Explicitly include the quotes in which you reached your conclusion in your output. 
    * If you cannot find any quotes or evidence to suggest that the lab offers analytical chemistry services, your answer should be a clear "NO".
    * If all information about a lab is overwhelmingly not close to analytical chemistry services, your answer should be a clear "NO" too. 
    * If you cannot find the home page of this lab, your answer should be a clear "NO" because the lab may no longer be active. Be sure to include the homepage URL in your output.
    * If some resources suggests that the lab is closed or suspended, your answer should be a clear "NO".
    * If you have neither enough evidence to believe this lab offers analytical chemistry services nor enough evidence to rule out the possibility, just say "MAYBE".
    
    ## Output format

    Respond with the following numbered list with each item on a new line (do not break lines for each item):
    1. YES, NO, or MAYBE
    2. Brief explanation of your reasoning including quotes from the information source to justify your reasoning
    3. Quotes from the information source to justify your reasoning  -- can be multiple quotes, separated by semicolons
    4. Types of tests offered by the lab (if available) -- extract this from the information source; respond with a list of tests separated by commas
    5. Industries the lab serves (if available) -- extract this from the information source; respond with a list of industries separated by commas
    6. Homepage URL of the lab (if available) -- just the top-level URL, no need to be pages specific to analytical chemistry services.

    ### Example output 1 
    1. YES
    2. The lab offers some service that are related to analytical chemistry.
    3. "We offer a wide range of analytics for chemicals"
    4. mass spectrometry, chromatography, spectroscopy, etc.
    5. Food and pharmaceuticals
    6. Homepage URL: https://www.example.com/

    ### Example output 2
    1. NO
    2. The lab homepage says they only test the safety of car seats.
    3. "We dedicate to testing the safety of car seats."
    4. car seat safety testing
    5. Automotive
    6. Homepage URL: https://www.example.com/

    ### Example output 3
    1. MAYBE
    2. The quote says that they test the metal hardness but it does not say whether this is the only service they offer. They may also do other services that are related to analytical chemistry.
    3. "We test the metal hardness."
    4. metal hardness testing
    5. Construction
    6. Homepage URL: https://www.example.com/

    ### Example output 4
    1. NO
    2. Cannot find the homepage of this lab. It may be inactive.
    3. N/A
    4. N/A
    5. N/A
    6. Homepage URL: N/A

    ## Lab Information:
    {lab_info}
    """

    client, model_name, config = create_gemini_client(model_version, use_search=True)
    config["system_instruction"] = system_prompt
    
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt.format(lab_info=lab_info),
        config=config
    )
    return response

def calculate_cost(response, model_id):
    web_search_queries = response.candidates[0].grounding_metadata.web_search_queries if "grounding_metadata" in response.candidates[0] else []

    pricing = { # per 1M tokens
        "gemini-2.5-flash": 
        {"input": 0.30, "output": 2.5, "cache": 0.075, "web_search": 0.035},
        "gemini-2.5-pro":
        {"input": 1.25, "output": 10, "cache": 0.31, "web_search": 0.035},
    }

    pricing_for_model = pricing[model_id]

    cost = 0
    cost += pricing_for_model["input"] * response.usage_metadata.prompt_token_count / 1000000
    cost += pricing_for_model["output"] * (response.usage_metadata.candidates_token_count + response.usage_metadata.thoughts_token_count) / 1000000
    num_cached_tokens = 0 if response.usage_metadata.cached_content_token_count is None else response.usage_metadata.cached_content_token_count
    cost += pricing_for_model["cache"] * num_cached_tokens / 1000000
    cost += pricing_for_model["web_search"] * len(web_search_queries) / 1000

    return cost

def extract_url_from_response(response_text: str) -> str:
    """Extract URL from the response text using regex."""
    if not response_text:
        return ''
    
    # Look for URLs in the response
    url_pattern = r'https?://[^\s\)\],]+'
    url_matches = re.findall(url_pattern, response_text)
    
    # Return the first URL found, or empty string if none
    return url_matches[0] if url_matches else ''

def majority_vote_urls(urls: list) -> str:
    """Determine the majority vote from a list of URLs."""
    # Filter out empty URLs
    valid_urls = [url for url in urls if url and url.strip()]
    
    if not valid_urls:
        return ''
    
    # Count occurrences of each URL
    url_counts = Counter(valid_urls)
    
    # Return the most common URL
    most_common = url_counts.most_common(1)
    return most_common[0][0] if most_common else ''

def find_website_url(lab_info: str, model_version="flash"):
    """Find the website URL for the lab/business using majority voting from 3 attempts."""
    
    system_prompt = """
    You are a research assistant helping to find website URLs for companies/labs.
    """
    
    user_prompt = """
    Please search for the only official website URL of this laboratory/business.
    
    ## Your Task
    Find the official website URL for this lab/company by searching for:
    1. Lab name + "website"
    2. Lab name + "official site"
    3. Lab name + address (if available)
    4. Lab name + company info
    
    ## Search Strategy
    - Use multiple search queries with different combinations
    - Look for official company websites, not third-party directories
    - Prioritize the main company website over specific service pages
    - Include the only one most relevant official website URL in your output
    
    ## Output format
    Respond with a numbered list:
    1. Website URL (if found) or "N/A" if not found
    2. Brief explanation of how you found it or why it wasn't found
    
    ## Lab Information:
    {lab_info}
    """
    
    client, model_name, config = create_gemini_client(model_version, use_search=True)
    config["system_instruction"] = system_prompt
    
    # Perform 3 attempts for majority voting
    urls = []
    responses = []
    for _ in range(3):
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt.format(lab_info=lab_info),
            config=config
        )
        responses.append(response)
        url = extract_url_from_response(response.text)
        urls.append(url)
    
    # Return majority vote result
    return majority_vote_urls(urls), responses

def check_website_active(lab_info: str, website_url: str, website_content: dict, model_version="flash"):
    """Check if the website is active and contains relevant business information."""
    
    system_prompt = """
    You are a research assistant helping to verify if company websites are active and contain relevant business information.
    """
    
    user_prompt = """
    Please analyze this laboratory's website to determine if it's active and contains information that aligns with the business query.
    
    ## Website Information Provided:
    - Website URL: {website_url}
    - HTTP Status Code: {status_code}
    - Website Content: {website_content}
    
    ## Your Task
    Determine if this website is:
    1. Active and accessible (loads properly)
    2. Contains current/recent business information 
    3. Shows the company is still in operation
    4. Has content that matches the lab information provided
    
    ## Decision Criteria
    - Answer "YES" if: Website loads, shows recent activity, and contains relevant business info
    - Answer "NO" if: Website doesn't load, shows very outdated info (>2 years), or company appears closed
    - Answer "MAYBE" if: Website loads but unclear if still active or mixed signals
    
    ## Output format
    Respond with a numbered list:
    1. YES, NO, or MAYBE
    2. Brief explanation with evidence from website analysis
    3. Last updated/activity indicators found (if any)
    4. Alignment between lab info and website content (if determinable)
    
    ## Lab Information:
    {lab_info}
    """
    
    # Prepare website content for analysis
    status_code = website_content.get('status_code', 0)
    content = website_content.get('content', 'No content available')[:5000]  # Limit content
    
    client, model_name, config = create_gemini_client(model_version)
    config["system_instruction"] = system_prompt
    
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt.format(
            website_url=website_url,
            status_code=status_code,
            website_content=content,
            lab_info=lab_info
        ),
        config=config
    )
    return response

def filter_chemistry_industry(lab_info: str, model_version="flash"):
    """Filter to check if the lab serves chemistry-related industries."""
    
    system_prompt = """
    You are a research assistant helping to identify if laboratories serve chemistry-related industries.
    """
    
    user_prompt = """
    Please research this laboratory to determine if it serves chemistry-related industries and its primary focus is on chemistry-related industries.
    
    ## Chemistry-related industries include:
    - Chemical manufacturing
    - Pharmaceuticals
    - Food and beverage testing
    - Environmental testing
    - Materials science
    - Petrochemicals
    - Cosmetics
    - Agriculture/fertilizers
    - Mining and metals
    - Water treatment
    - Research institutions
    
    ## Non-chemistry industries (that we want to filter out if the lab primarily serves industries that are not chemistry-related):
    - **Medical/clinical testing**
    - **Construction testing**
    - **Biological/microbiological testing**
    - Mechanical testing
    - Electrical testing
    - Software/IT services
    - Automotive testing
    - Telecommunications
    - Financial services
    
    ## Your Task
    1. Determine if this lab primarily serves chemistry-related industries.
    2. If this lab serves primarily on chemistry-related industries, determine the percentage of chemistry-related vs non-chemistry work.
    3. If the percentage of chemistry-related work is 100%, answer "YES". Otherwise, answer "MAYBE".
    
    ## Information sources to use
    1. The lab info provided below
    2. Lab website content
    3. Web search results about this lab and its services
    
    ## Search Strategy
    1. Search for lab name + "industries served"
    2. Search for lab name + "clients" or "customers"
    3. Search for lab name + "services" to understand what they offer
    4. Look for case studies or testimonials that indicate industry focus
    
    ## Decision Criteria
    - Answer "YES" if: Lab primarily serves industries directly related to chemistry
    - Answer "NO" if: Lab primarily serves industries not related to chemistry
    - Answer "MAYBE" if: Mixed industries, not directly related to chemistry, or unclear focus
    
    ## Output format
    Respond with a numbered list:
    1. YES, NO, or MAYBE
    2. Brief explanation with evidence
    3. Primary industries served (list)
    4. Percentage of chemistry-related work out of all work (related + non-related), just a percentage number and sign, no other text without new line
    
    ## Lab Information:
    {lab_info}
    """
    
    client, model_name, config = create_gemini_client(model_version, use_search=True)
    config["system_instruction"] = system_prompt
    
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt.format(lab_info=lab_info),
        config=config
    )
    return response

def filter_commercial_testing(lab_info: str, model_version="flash"):
    """Filter to check if the lab offers commercial analytical testing services (not just internal testing)."""
    
    system_prompt = """
    You are a research assistant helping to identify if laboratories offer commercial analytical testing services to external clients.
    """
    
    user_prompt = """
    Please research this laboratory to determine if it offers commercial analytical testing services to external clients, or if it only conducts internal testing for its own products/research.
    
    ## Commercial analytical testing indicators:
    - Accepts samples from external clients
    - Lists testing services on website for hire
    - Has pricing or quotes for testing services
    - Mentions "contract testing" or "third-party testing"
    - Has client testimonials from external companies
    - Advertises analytical services to other businesses
    - Has accreditations for commercial testing (ISO 17025, etc.)
    
    ## Internal testing only indicators:
    - Only tests their own products
    - In-house R&D lab for single company
    - Quality control lab for manufacturing
    - Research lab that doesn't accept external samples
    - University research lab (unless explicitly offering commercial services)
    - No mention of external clients or commercial services
    
    ## Your Task
    Determine if this lab offers commercial analytical testing services to external clients or only does internal testing.
    
    ## Information sources to use
    1. The lab info provided below
    2. Web search results about their services
    3. Lab website, especially services/pricing pages
    4. Customer testimonials or case studies
    
    ## Search Strategy
    1. Search for lab name + "testing services" + "commercial"
    2. Search for lab name + "contract testing" or "third party testing"
    3. Search for lab name + "clients" to see if they mention external customers
    4. Look for accreditations or certifications for commercial testing
    
    ## Decision Criteria
    - Answer "YES" if: Clear evidence of commercial testing services for external clients
    - Answer "NO" if: Only internal testing or no evidence of external commercial services
    - Answer "MAYBE" if: Unclear or mixed evidence
    
    ## Output format
    Respond with a numbered list:
    1. YES, NO, or MAYBE
    2. Brief explanation with evidence
    3. Types of commercial services offered (if any)
    4. Evidence of external clients (quotes/testimonials if found)
    
    ## Lab Information:
    {lab_info}
    """
    
    client, model_name, config = create_gemini_client(model_version, use_search=True)
    config["system_instruction"] = system_prompt
    
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt.format(lab_info=lab_info),
        config=config
    )
    return response

def apply_website_url_filter(df, output_file, model_version="flash"):
    """Stage 1: Find website URLs for labs."""
    # Load existing results and initialize output file
    processed_lab_ids, results = load_existing_results(output_file, required_key='website_url')
    initialize_output_file(output_file, len(df))
    
    # Filter out already processed labs
    unprocessed_labs = []
    for index, lab in df.iterrows():
        lab_id = lab.get('id', index)
        if lab_id not in processed_lab_ids:
            unprocessed_labs.append((index, lab))
    
    print(f"Stage 1: Finding website URLs for {len(unprocessed_labs)} labs (skipping {len(processed_lab_ids)} already processed)")
    
    for idx, (index, lab) in enumerate(tqdm(unprocessed_labs, desc="Finding website URLs")):
        lab_id = lab.get('id', index)
        print(f"Processing lab {idx + 1}/{len(unprocessed_labs)} (ID: {lab_id}): {lab.get('name', 'Unknown')}")
        
        # Prepare lab info and initialize lab dictionary
        lab_info = prepare_lab_info(lab)
        lab_dict = lab.to_dict()
        lab_dict['id'] = lab_id
        
        # Find website URL
        print("  Finding website URL...")
        website_url, url_responses = find_website_url(lab_info, model_version)
        url_cost = sum(calculate_cost(response, f"gemini-2.5-{model_version}") for response in url_responses)
        
        lab_dict['website_url'] = website_url
        lab_dict['website_url_cost'] = url_cost
        lab_dict['website_url_status'] = 'YES' if website_url != '' else 'NO'
        
        print(f"  Website URL: {website_url if website_url else 'Not found'}")
        print(f"  Cost: ${url_cost:.4f}")
        
        results.append(lab_dict)
        save_results_incrementally(results, output_file, idx, len(unprocessed_labs))
    
    return results

def apply_industry_filter(df, output_file, model_version="flash"):
    """Stage 2: Check if labs serve chemistry-related industries."""
    # Load existing results and initialize output file
    processed_lab_ids, results = load_existing_results(output_file, required_key='chemistry_industry')
    initialize_output_file(output_file, len(df))
    
    # Filter out already processed labs
    unprocessed_labs = []
    for index, lab in df.iterrows():
        lab_id = lab.get('id', index)
        if lab_id not in processed_lab_ids:
            unprocessed_labs.append((index, lab))
    
    print(f"Stage 2: Checking chemistry industry for {len(unprocessed_labs)} labs (skipping {len(processed_lab_ids)} already processed)")
    
    for idx, (index, lab) in enumerate(tqdm(unprocessed_labs, desc="Checking chemistry industry")):
        lab_id = lab.get('id', index)
        print(f"Processing lab {idx + 1}/{len(unprocessed_labs)} (ID: {lab_id}): {lab.get('name', 'Unknown')}")
        
        # Initialize lab dictionary
        lab_dict = lab.to_dict()
        lab_dict['id'] = lab_id
        
        # Check if website URL exists (from previous stage)
        website_url = lab.get('website_url', '')
        
        if website_url != '':
            # Prepare lab info and check chemistry industry relevance
            lab_info = prepare_lab_info(lab)
            
            print("  Checking chemistry industry relevance...")
            
            # Retry up to 3 times if we can't extract the percentage
            max_retries = 3
            percentage = 0
            industry_decision = ""
            
            for attempt in range(max_retries):
                industry_result = filter_chemistry_industry(lab_info, model_version)
                industry_cost = calculate_cost(industry_result, f"gemini-2.5-{model_version}")
                industry_text = industry_result.text
                
                # Parse industry filter result using new abstraction
                parsed_items = parse_numbered_response(industry_text, 4)
                industry_decision = parsed_items[0]  # Line 1
                percentage_line = parsed_items[3]    # Line 4
                
                # Extract percentage
                if percentage_line:
                    percentage_match = re.search(r'(\d+)%', percentage_line)
                    if percentage_match:
                        percentage = int(percentage_match.group(1))
                        break  # Successfully extracted percentage, exit retry loop
                
                if attempt < max_retries - 1:
                    print(f"    Failed to extract percentage on attempt {attempt + 1}, retrying...")
                else:
                    print(f"    Failed to extract percentage after {max_retries} attempts, using 0%")
            
            # Update logic: set percentages >= 50% as YES
            # if 'YES' in industry_decision.upper() or percentage >= 50:
            if 'YES' in industry_decision.upper():
                lab_dict['chemistry_industry'] = 'YES'
            elif 'MAYBE' in industry_decision.upper():
                lab_dict['chemistry_industry'] = 'MAYBE'
            else:
                lab_dict['chemistry_industry'] = 'NO'
            
            lab_dict['chemistry_percentage'] = percentage
            lab_dict['industry_filter_details'] = industry_text
            lab_dict['chemistry_industry_cost'] = industry_cost
            
            print(f"  Chemistry industry: {lab_dict['chemistry_industry']}")
            print(f"  Cost: ${industry_cost:.4f}")
        else:
            # Skip this filter if no website URL
            lab_dict['chemistry_industry'] = 'SKIPPED'
            lab_dict['industry_filter_details'] = 'SKIPPED - no website URL found'
            lab_dict['chemistry_industry_cost'] = 0
            print("  Skipped - no website URL found")
        
        results.append(lab_dict)
        save_results_incrementally(results, output_file, idx, len(unprocessed_labs))
    
    return results

def filter_chemistry_labs(df, output_file, model_version="pro"):
    """Filter labs to find those offering analytical chemistry services."""
    # Load existing results and initialize output file
    processed_lab_ids, results = load_existing_results(output_file)
    initialize_output_file(output_file, len(df))
    
    # Filter out already processed labs
    unprocessed_labs = []
    for index, lab in df.iterrows():
        lab_id = lab.get('id', index)
        if lab_id not in processed_lab_ids:
            unprocessed_labs.append((index, lab))
    
    print(f"Total labs to process: {len(unprocessed_labs)} (skipping {len(processed_lab_ids)} already processed)")
    
    for idx, (index, lab) in enumerate(tqdm(unprocessed_labs, desc="Researching labs")):
        lab_id = lab.get('id', index)
        print(f"Researching lab {idx + 1}/{len(unprocessed_labs)} (ID: {lab_id}): {lab.get('name', 'Unknown')}")
        
        # Check if lab passed prefilters
        has_website_url = (lab.get('website_url', '') != '')
        has_chemistry_industry = (lab.get('chemistry_industry') == 'YES')
        
        # Create lab entry
        lab_dict = lab.to_dict()
        lab_dict['id'] = lab_id
        
        if not (has_website_url and has_chemistry_industry):
            # Skip deep filter for labs without website URL
            _set_skipped_lab_data(lab_dict, 'Skipped due to failed prefilters')
            print(f"  Skipped lab (failed prefilters): {lab.get('name', 'Unknown')}")
        else:
            # Run deep filter for labs with website URL
            exclude_columns = ['id', 'logo url', 'standards', 'qualifications', 'gallery', 'publications', 'website_content', 'website_filter_details']
            lab_info = prepare_lab_info(lab, exclude_columns)

            # Include website content if available
            enhanced_lab_info = lab_info
            website_content = lab.get('website_content', {})
            if website_content is not None and isinstance(website_content, dict) and website_content.get('status_code') == 200 and website_content.get('content'):
                enhanced_lab_info += f"\n\nWebsite Content:\n{website_content['content'][:8000]}..."
            else:
                _set_skipped_lab_data(lab_dict, 'Skipped due to no website content')

            research_result = research_lab(enhanced_lab_info, model_version)
            cost = calculate_cost(research_result, f"gemini-2.5-{model_version}")
            print(f"Cost: {cost}")

            lab_dict['cost'] = cost
            _parse_research_result(lab_dict, research_result.text, lab)
        
        lab_dict.pop('logo_url', None)
        results.append(lab_dict)
        save_results_incrementally(results, output_file, idx, len(unprocessed_labs))
    
    return results

def _set_skipped_lab_data(lab_dict, reason):
    """Set standard data for skipped labs."""
    lab_dict['is_chemistry_lab'] = 'SKIPPED'
    lab_dict['cost'] = 0
    lab_dict['research_reason'] = reason
    lab_dict['research_quotes'] = ''
    lab_dict['test_types'] = ''
    lab_dict['industries_served'] = ''
    lab_dict['homepage_url'] = ''

def _parse_research_result(lab_dict, research_text, lab):
    """Parse research result and populate lab dictionary."""
    try:
        # Parse numbered response using abstraction
        parsed_items = parse_numbered_response(research_text, 6)
        
        # Determine lab qualification status
        decision_text = parsed_items[0].upper()
        if 'YES' in decision_text:
            lab_dict['is_chemistry_lab'] = 'YES'
            print(f"✓ Lab qualified: {lab.get('name', 'Unknown')}")
        elif 'MAYBE' in decision_text:
            lab_dict['is_chemistry_lab'] = 'MAYBE'
            print(f"? Lab maybe qualified: {lab.get('name', 'Unknown')}")
        else:
            lab_dict['is_chemistry_lab'] = 'NO'
            print(f"✗ Lab not qualified: {lab.get('name', 'Unknown')}")
        
        # Extract other fields
        lab_dict['research_reason'] = parsed_items[1]
        lab_dict['research_quotes'] = parsed_items[2]
        
        # Clean test types (remove prefix if present)
        tests_text = parsed_items[3]
        if 'Types of tests offered by the lab:' in tests_text:
            tests_text = tests_text.split('Types of tests offered by the lab:')[1].strip()
        lab_dict['test_types'] = tests_text
        
        # Clean industries (remove prefix if present)
        industries_text = parsed_items[4]
        if 'Industries the lab serves:' in industries_text:
            industries_text = industries_text.split('Industries the lab serves:')[1].strip()
        lab_dict['industries_served'] = industries_text
        
        # Extract URL
        url_text = parsed_items[5]
        if 'Homepage URL:' in url_text:
            extracted_url = url_text.split('Homepage URL:')[1].strip()
        elif 'URL:' in url_text:
            extracted_url = url_text.split('URL:')[1].strip()
        else:
            extracted_url = url_text if url_text.startswith(('http://', 'https://', 'www.')) or 'N/A' in url_text else ''
        lab_dict['homepage_url'] = extracted_url
        
    except Exception as e:
        print(f"Warning: Error parsing research result: {e}")
        # Fallback parsing
        lab_dict['research_reason'] = research_text[:200] + '...' if research_text and len(research_text) > 200 else research_text
        lab_dict['research_quotes'] = ''
        lab_dict['test_types'] = ''
        lab_dict['industries_served'] = ''
        lab_dict['homepage_url'] = ''
        
        # Fallback qualification determination
        if research_text and "YES" in research_text.upper():
            lab_dict['is_chemistry_lab'] = 'YES'
        elif research_text and "MAYBE" in research_text.upper():
            lab_dict['is_chemistry_lab'] = 'MAYBE'
        else:
            lab_dict['is_chemistry_lab'] = 'NO'

def save_results(qualified_labs, output_filename):
    """Save qualified labs to output CSV file."""
    if not qualified_labs:
        print("No qualified labs found.")
        return
    
    df_output = pd.DataFrame(qualified_labs)
    df_output.to_csv(output_filename, index=False)
    print(f"Saved {len(qualified_labs)} qualified labs to {output_filename}")

def main(num_samples=20, seed=42, model_version="pro", input_file="input_labs.json", output_file="output_labs.json", start_stage=1, end_stage=3):
    """Main function to orchestrate the lab filtering process with stage-based processing."""
    
    # Generate intermediate file names
    base_name = output_file.replace('.json', '')
    stage1_file = f"{base_name}_stage1_website_urls.json"
    stage2_file = f"{base_name}_stage2_industry_filter.json"
    
    print(f"Processing stages {start_stage} to {end_stage}")
    
    # Determine starting dataframe based on start_stage
    if start_stage == 1:
        # Load input JSON
        df = load_labs_json(input_file)
        if df is None:
            return
    elif start_stage == 2:
        # Load stage 1 results
        df = load_labs_json(stage1_file)
        if df is None:
            print(f"Error: Stage 1 results not found at {stage1_file}")
            return
    elif start_stage == 3:
        # Load stage 2 results
        df = load_labs_json(stage2_file)
        if df is None:
            print(f"Error: Stage 2 results not found at {stage2_file}")
            return
    
    # Test mode: randomly sample num_samples rows (only if starting from stage 1)
    if start_stage == 1 and num_samples > 0:
        print(f"Sampling {num_samples} random labs to process")
        df = df.sample(n=min(num_samples, len(df)), random_state=seed)
        df.reset_index(drop=True, inplace=True)
    elif start_stage == 1:
        # shuffle rows
        df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Using model: gemini-2.5-{model_version}")
    
    current_results = df
    
    # Stage 1: Website URL finding
    if start_stage <= 1 and end_stage >= 1:
        print(f"\n=== STAGE 1: Finding Website URLs ===")
        print(f"Results will be saved to: {stage1_file}")
        current_results = apply_website_url_filter(current_results, stage1_file, "flash")
        
        # Print stage 1 statistics
        website_found = len([lab for lab in current_results if lab.get('website_url_status') == 'YES'])
        website_not_found = len([lab for lab in current_results if lab.get('website_url_status') == 'NO'])
        print(f"\n=== Stage 1 Results ===")
        print(f"Total labs processed: {len(current_results)}")
        print(f"Website URLs found: {website_found} ({website_found/len(current_results)*100:.1f}%)")
        print(f"Website URLs not found: {website_not_found} ({website_not_found/len(current_results)*100:.1f}%)")
        
        if end_stage == 1:
            return current_results
        
        # Convert to DataFrame for next stage
        current_results = pd.DataFrame(current_results)
    
    # Stage 2: Chemistry industry filtering
    if start_stage <= 2 and end_stage >= 2:
        print(f"\n=== STAGE 2: Chemistry Industry Filtering ===")
        print(f"Results will be saved to: {stage2_file}")
        current_results = apply_industry_filter(current_results, stage2_file, "flash")
        
        # Print stage 2 statistics
        industry_yes = len([lab for lab in current_results if lab.get('chemistry_industry') == 'YES'])
        industry_no = len([lab for lab in current_results if lab.get('chemistry_industry') == 'NO'])
        industry_maybe = len([lab for lab in current_results if lab.get('chemistry_industry') == 'MAYBE'])
        industry_skipped = len([lab for lab in current_results if lab.get('chemistry_industry') == 'SKIPPED'])
        
        print(f"\n=== Stage 2 Results ===")
        print(f"Total labs processed: {len(current_results)}")
        print(f"Chemistry industry YES: {industry_yes} ({industry_yes/len(current_results)*100:.1f}%)")
        print(f"Chemistry industry NO: {industry_no} ({industry_no/len(current_results)*100:.1f}%)")
        print(f"Chemistry industry MAYBE: {industry_maybe} ({industry_maybe/len(current_results)*100:.1f}%)")
        print(f"Chemistry industry SKIPPED: {industry_skipped} ({industry_skipped/len(current_results)*100:.1f}%)")
        
        # Summary of labs passing prefilters
        passed_prefilters = [lab for lab in current_results if 
                           lab.get('website_url_status') == 'YES' and 
                           lab.get('chemistry_industry') == 'YES']
        print(f"Labs passing all prefilters: {len(passed_prefilters)} ({len(passed_prefilters)/len(current_results)*100:.1f}%)")
        
        if end_stage == 2:
            return current_results
        
        # Convert to DataFrame for next stage
        current_results = pd.DataFrame(current_results)
    
    # Stage 3: Chemistry lab research (final stage)
    if start_stage <= 3 and end_stage >= 3:
        print(f"\n=== STAGE 3: Chemistry Lab Research ===")
        print(f"Results will be saved to: {output_file}")
        
        # Apply chemistry lab filtering
        chemistry_results = filter_chemistry_labs(current_results, output_file, "pro")
        
        print(f"\n=== Final Results ===")
        print(f"Chemistry research completed. Results saved to {output_file}")
        chemistry_labs = [lab for lab in chemistry_results if lab.get('is_chemistry_lab') == 'YES']
        maybe_labs = [lab for lab in chemistry_results if lab.get('is_chemistry_lab') == 'MAYBE']
        
        print(f"Final qualified chemistry labs: {len(chemistry_labs)} out of {len(current_results)} original labs ({len(chemistry_labs)/len(current_results)*100:.1f}%)")
        print(f"Maybe chemistry labs: {len(maybe_labs)} out of {len(current_results)} original labs ({len(maybe_labs)/len(current_results)*100:.1f}%)")
        print(f"Total labs that passed all filters: {len(chemistry_labs) + len(maybe_labs)} out of {len(current_results)} original labs")
        
        # Save "yes_only" output file containing only YES labs
        yes_only_file = output_file.replace('.json', '_yes_only.json')
        if chemistry_labs:
            with open(yes_only_file, 'w', encoding='utf-8') as f:
                json.dump(chemistry_labs, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(chemistry_labs)} YES-qualified labs to {yes_only_file}")
        else:
            print("No YES-qualified labs found - no yes_only file created")
        
        return chemistry_results
    
    return current_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find analytical chemistry labs')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to process. Set to 0 to process all.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--version', choices=['pro', 'flash'], default='pro', 
                        help='Select Gemini model version: "pro" for gemini-2.5-pro or "flash" for gemini-2.5-flash')
    parser.add_argument('--input', default='input_labs.json',
                            help='Input JSON file name (default: input_labs.json)')
    parser.add_argument('--output', default='output_labs.json',
                        help='Output JSON file name (default: output_labs.json)')
    parser.add_argument('--start-stage', type=int, choices=[1, 2, 3], default=1,
                        help='Starting stage: 1=website URLs, 2=industry filter, 3=chemistry research')
    parser.add_argument('--end-stage', type=int, choices=[1, 2, 3], default=3,
                        help='Ending stage: 1=website URLs, 2=industry filter, 3=chemistry research')
    
    args = parser.parse_args()
    main(num_samples=args.num_samples, seed=args.seed, model_version=args.version, input_file=args.input, output_file=args.output, start_stage=args.start_stage, end_stage=args.end_stage)
