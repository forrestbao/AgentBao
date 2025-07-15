import os
import json

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from google import genai
from google.genai import types
from google.genai.types import Tool, GoogleSearch


load_dotenv()

def load_labs_json(filename):
    """Load and parse the input JSON file containing lab information."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to DataFrame with index
        df = pd.DataFrame.from_dict(data, orient='index')
        print(f"Loaded {len(df)} labs from {filename}")
        print(f"Index: {df.index.name or 'Index'}")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def research_lab(lab_info: str, model_version="pro"):

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

    client = genai.Client()

    google_search_tool = Tool(
        google_search = GoogleSearch()
    )

    # Select model based on version
    model_name = f"gemini-2.5-{model_version}"
    
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt.format(lab_info=lab_info),
        config= {
           "tools": [google_search_tool],
           "response_modalities": ["TEXT"],
           "thinking_config": types.ThinkingConfig(thinking_budget=-1), 
           "system_instruction": system_prompt
        }
    )
    return response

def calculate_cost(response, model_id):
    web_search_queries = response.candidates[0].grounding_metadata.web_search_queries

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

def filter_website_exists(lab_info: str, model_version="pro"):
    """Filter to check if the lab's website exists and the company is still active."""
    
    system_prompt = """
    You are a research assistant helping to verify if companies/labs are still active and have functional websites.
    """
    
    user_prompt = """
    Please research this laboratory to determine if it has a functional website and appears to still be in business.
    
    ## Your Task
    Check if this lab/company:
    1. Has a working, accessible website
    2. Appears to still be active/in business (not closed, out of business, or suspended)
    3. The website loads properly and contains current information
    
    ## Information sources to use
    1. The lab info provided below
    2. Web search results about this lab
    3. Direct website visits (if website URLs are found)
    
    ## Search Strategy
    1. Search for the lab name + "website"
    2. Search for the lab name + address to find current status
    3. Search for the lab name + "closed" or "out of business" to check if it's still operating
    4. Try to access any found websites directly
    
    ## Decision Criteria
    - Answer "YES" if: Website exists, loads properly, and shows current/recent activity
    - Answer "NO" if: No website found, website doesn't load, company appears closed/out of business, or website shows very outdated information (>2 years old)
    - Answer "MAYBE" if: Unclear status or mixed signals about whether company is active
    
    ## Output format
    Respond with a numbered list:
    1. YES, NO, or MAYBE
    2. Brief explanation with evidence from your research
    3. Website URL (if found and working) or "N/A"
    4. Last updated/activity date (if determinable) or "N/A"
    
    ## Lab Information:
    {lab_info}
    """
    
    client = genai.Client()
    google_search_tool = Tool(google_search = GoogleSearch())
    model_name = f"gemini-2.5-{model_version}"
    
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt.format(lab_info=lab_info),
        config= {
           "tools": [google_search_tool],
           "response_modalities": ["TEXT"],
           "thinking_config": types.ThinkingConfig(thinking_budget=-1), 
           "system_instruction": system_prompt
        }
    )
    return response

def filter_chemistry_industry(lab_info: str, model_version="pro"):
    """Filter to check if the lab serves chemistry-related industries."""
    
    system_prompt = """
    You are a research assistant helping to identify if laboratories serve chemistry-related industries.
    """
    
    user_prompt = """
    Please research this laboratory to determine if it serves chemistry-related industries or if its primary focus is on non-chemistry sectors.
    
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
    
    ## Non-chemistry industries (that we want to filter out):
    - Pure medical/clinical testing (without chemical analysis)
    - Mechanical testing only
    - Electrical testing only
    - Software/IT services
    - Construction testing (structural only)
    - Automotive testing (safety/mechanical only)
    - Telecommunications
    - Financial services
    - Pure biological/microbiological testing
    
    ## Your Task
    Determine if this lab primarily serves chemistry-related industries or non-chemistry industries.
    
    ## Information sources to use
    1. The lab info provided below
    2. Web search results about this lab and its services
    3. Lab website content (if available)
    
    ## Search Strategy
    1. Search for lab name + "industries served"
    2. Search for lab name + "clients" or "customers"
    3. Search for lab name + "services" to understand what they offer
    4. Look for case studies or testimonials that indicate industry focus
    
    ## Decision Criteria
    - Answer "YES" if: Lab primarily serves chemistry-related industries
    - Answer "NO" if: Lab primarily serves non-chemistry industries
    - Answer "MAYBE" if: Mixed industries or unclear focus
    
    ## Output format
    Respond with a numbered list:
    1. YES, NO, or MAYBE
    2. Brief explanation with evidence
    3. Primary industries served (list)
    4. Percentage estimate of chemistry-related vs non-chemistry work (if determinable)
    
    ## Lab Information:
    {lab_info}
    """
    
    client = genai.Client()
    google_search_tool = Tool(google_search = GoogleSearch())
    model_name = f"gemini-2.5-{model_version}"
    
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt.format(lab_info=lab_info),
        config= {
           "tools": [google_search_tool],
           "response_modalities": ["TEXT"],
           "thinking_config": types.ThinkingConfig(thinking_budget=-1), 
           "system_instruction": system_prompt
        }
    )
    return response

def filter_commercial_testing(lab_info: str, model_version="pro"):
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
    
    client = genai.Client()
    google_search_tool = Tool(google_search = GoogleSearch())
    model_name = f"gemini-2.5-{model_version}"
    
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt.format(lab_info=lab_info),
        config= {
           "tools": [google_search_tool],
           "response_modalities": ["TEXT"],
           "thinking_config": types.ThinkingConfig(thinking_budget=-1), 
           "system_instruction": system_prompt
        }
    )
    return response

def apply_pre_filters(df, output_file, model_version="pro"):
    """Apply the three pre-filters before chemistry lab filtering."""
    results = []
    
    # Check for existing results to continue from interruption
    processed_lab_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                for lab_entry in existing_results:
                    if 'id' in lab_entry:
                        processed_lab_ids.add(lab_entry['id'])
                results = existing_results
                print(f"Found {len(processed_lab_ids)} already processed labs in {output_file}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not read existing results from {output_file}: {e}")
            print("Starting fresh...")
    
    # Initialize output JSON file if it doesn't exist
    if len(df) > 0 and not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
    
    # Filter out already processed labs
    unprocessed_labs = []
    for index, lab in df.iterrows():
        lab_id = lab.get('id', index)
        if lab_id not in processed_lab_ids:
            unprocessed_labs.append((index, lab))
    
    print(f"Applying pre-filters to {len(unprocessed_labs)} labs (skipping {len(processed_lab_ids)} already processed)")
    
    for idx, (index, lab) in enumerate(tqdm(unprocessed_labs, desc="Applying pre-filters")):
        lab_id = lab.get('id', index)
        print(f"Pre-filtering lab {idx + 1}/{len(unprocessed_labs)} (ID: {lab_id}): {lab.get('name', 'Unknown')}")
        
        # Prepare lab info string
        lab_info = ""
        do_not_use_column = ['id', 'logo url', 'standards', 'qualifications', 'gallery', 'publications']
        for k, v in lab.to_dict().items():
            if (pd.notna(v) and 
                str(v).strip().lower() not in ['', 'nan', 'none'] and 
                k not in do_not_use_column):
                lab_info += f"{k}: {v}, "
        
        # Initialize lab dictionary and cost tracking
        lab_dict = lab.to_dict()
        lab_dict['id'] = lab_id
        total_cost = 0
        
        # Apply filters sequentially, stopping on first NO
        filter_failed = False
        failed_filter = ""
        
        # Filter 1: Website exists
        print("  Checking if website exists...")
        website_result = filter_website_exists(lab_info, model_version)
        website_cost = calculate_cost(website_result, f"gemini-2.5-{model_version}")
        website_text = website_result.text
        total_cost += website_cost
        
        # Parse website filter result
        website_lines = website_text.split('\n') if website_text else []
        website_decision = next((line for line in website_lines if line.strip().startswith('1.')), '')
        if 'YES' in website_decision.upper():
            lab_dict['website_exists'] = 'YES'
        elif 'MAYBE' in website_decision.upper():
            lab_dict['website_exists'] = 'MAYBE'
        else:
            lab_dict['website_exists'] = 'NO'
            filter_failed = True
            failed_filter = "website"
        lab_dict['website_filter_details'] = website_text
        
        # Filter 2: Chemistry industry (only if website filter passed)
        if not filter_failed:
            print("  Checking chemistry industry relevance...")
            industry_result = filter_chemistry_industry(lab_info, model_version)
            industry_cost = calculate_cost(industry_result, f"gemini-2.5-{model_version}")
            industry_text = industry_result.text
            total_cost += industry_cost
            
            # Parse industry filter result
            industry_lines = industry_text.split('\n') if industry_text else []
            industry_decision = next((line for line in industry_lines if line.strip().startswith('1.')), '')
            if 'YES' in industry_decision.upper():
                lab_dict['chemistry_industry'] = 'YES'
            elif 'MAYBE' in industry_decision.upper():
                lab_dict['chemistry_industry'] = 'MAYBE'
            else:
                lab_dict['chemistry_industry'] = 'NO'
                filter_failed = True
                failed_filter = "chemistry_industry"
            lab_dict['industry_filter_details'] = industry_text
        else:
            # Skip this filter
            lab_dict['chemistry_industry'] = 'SKIPPED'
            lab_dict['industry_filter_details'] = 'SKIPPED - failed previous filter'
        
        # Filter 3: Commercial testing (only if previous filters passed)
        if not filter_failed:
            print("  Checking commercial testing availability...")
            commercial_result = filter_commercial_testing(lab_info, model_version)
            commercial_cost = calculate_cost(commercial_result, f"gemini-2.5-{model_version}")
            commercial_text = commercial_result.text
            total_cost += commercial_cost
            
            # Parse commercial filter result
            commercial_lines = commercial_text.split('\n') if commercial_text else []
            commercial_decision = next((line for line in commercial_lines if line.strip().startswith('1.')), '')
            if 'YES' in commercial_decision.upper():
                lab_dict['commercial_testing'] = 'YES'
            elif 'MAYBE' in commercial_decision.upper():
                lab_dict['commercial_testing'] = 'MAYBE'
            else:
                lab_dict['commercial_testing'] = 'NO'
                filter_failed = True
                failed_filter = "commercial_testing"
            lab_dict['commercial_filter_details'] = commercial_text
        else:
            # Skip this filter
            lab_dict['commercial_testing'] = 'SKIPPED'
            lab_dict['commercial_filter_details'] = 'SKIPPED - failed previous filter'
        
        lab_dict['prefilter_cost'] = total_cost
        print(f"  Total pre-filter cost: ${total_cost:.4f}")
        
        # Determine if lab passes pre-filters
        passes_prefilters = (
            lab_dict['website_exists'] in ['YES'] and
            lab_dict['chemistry_industry'] in ['YES'] and
            lab_dict['commercial_testing'] in ['YES']
        )
        
        lab_dict['passes_prefilters'] = passes_prefilters
        
        if passes_prefilters:
            print(f"  ✓ Lab passed all pre-filters: {lab.get('name', 'Unknown')}")
        else:
            print(f"  ✗ Lab failed pre-filters at {failed_filter}: {lab.get('name', 'Unknown')}")
            print(f"    Website: {lab_dict['website_exists']}, Industry: {lab_dict['chemistry_industry']}, Commercial: {lab_dict['commercial_testing']}")
        
        results.append(lab_dict)
        
        # Save results after each lab
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved lab {idx + 1}/{len(unprocessed_labs)} to {output_file}")
    
    return results

def filter_chemistry_labs(df, output_file, model_version="pro"):
    """Filter labs to find those offering analytical chemistry services."""
    results = []
    
    # Check for existing results to continue from interruption
    processed_lab_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                # Extract IDs of already processed labs
                for lab_entry in existing_results:
                    if 'id' in lab_entry:
                        processed_lab_ids.add(lab_entry['id'])
                results = existing_results
                print(f"Found {len(processed_lab_ids)} already processed labs in {output_file}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not read existing results from {output_file}: {e}")
            print("Starting fresh...")
    
    # Initialize output JSON file with empty array if it doesn't exist
    if len(df) > 0 and not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
    
    # Filter out already processed labs
    unprocessed_labs = []
    for index, lab in df.iterrows():
        lab_id = lab.get('id', index)  # Use index as fallback if no ID column
        if lab_id not in processed_lab_ids:
            unprocessed_labs.append((index, lab))
    
    print(f"Total labs to process: {len(unprocessed_labs)} (skipping {len(processed_lab_ids)} already processed)")
    
    for idx, (index, lab) in enumerate(tqdm(unprocessed_labs, desc="Researching labs")):
        lab_id = lab.get('id', index)
        print(f"Researching lab {idx + 1}/{len(unprocessed_labs)} (ID: {lab_id}): {lab.get('name', 'Unknown')}")
        
        # Check if all prefilters are YES
        all_prefilters_yes = (
            lab.get('website_exists') == 'YES' and
            lab.get('chemistry_industry') == 'YES' and
            lab.get('commercial_testing') == 'YES'
        )
        
        # Create lab entry
        lab_dict = lab.to_dict()
        lab_dict['id'] = lab_id
        
        if not all_prefilters_yes:
            # Skip deep filter for labs without all YES prefilters
            lab_dict['is_chemistry_lab'] = 'SKIPPED'
            lab_dict['cost'] = 0
            lab_dict['research_reason'] = 'Skipped due to failed prefilters'
            lab_dict['research_quotes'] = ''
            lab_dict['test_types'] = ''
            lab_dict['industries_served'] = ''
            lab_dict['homepage_url'] = ''
            print(f"  Skipped lab (failed prefilters): {lab.get('name', 'Unknown')}")
        else:
            # Run deep filter for labs with all YES prefilters
            lab_info = ""
            do_not_use_column = ['id', 'logo url', 'standards', 'qualifications', 'gallery', 'publications']
            for k, v in lab.to_dict().items():
                # Skip empty values, None, NaN, 'nan' strings, and whitespace-only strings
                if (pd.notna(v) and 
                    str(v).strip().lower() not in ['', 'nan', 'none'] and 
                    k not in do_not_use_column):
                    lab_info += f"{k}: {v}, "

            research_result = research_lab(lab_info, model_version)
            model_id = f"gemini-2.5-{model_version}"
            cost = calculate_cost(research_result, model_id)
            print(f"Cost: {cost}")

            research_result = research_result.text
            lab_dict['cost'] = cost
            
            # Parse numbered list format from research result
            # Expected format:
            # 1. YES/NO/MAYBE
            # 2. Brief explanation of reasoning including quotes
            # 3. Quotes from the information source to justify reasoning
            # 4. Types of tests offered by the lab
            # 5. Industries the lab serves
            # 6. Homepage URL of the lab
            
            try:
                lines = research_result.split('\n') if research_result else []
                
                # Extract decision from line starting with "1."
                decision_line = next((line for line in lines if line.strip().startswith('1.')), '')
                if decision_line:
                    decision_text = decision_line.replace('1.', '').strip().upper()
                    if 'YES' in decision_text:
                        lab_dict['is_chemistry_lab'] = 'YES'
                        print(f"✓ Lab qualified: {lab.get('name', 'Unknown')}")
                    elif 'MAYBE' in decision_text:
                        lab_dict['is_chemistry_lab'] = 'MAYBE'
                        print(f"? Lab maybe qualified: {lab.get('name', 'Unknown')}")
                    else:
                        lab_dict['is_chemistry_lab'] = 'NO'
                        print(f"✗ Lab not qualified: {lab.get('name', 'Unknown')}")
                else:
                    # Fallback to old parsing if numbered format not found
                    if research_result and "YES" in research_result.upper():
                        lab_dict['is_chemistry_lab'] = 'YES'
                        print(f"✓ Lab qualified: {lab.get('name', 'Unknown')}")
                    elif research_result and "MAYBE" in research_result.upper():
                        lab_dict['is_chemistry_lab'] = 'MAYBE'
                        print(f"? Lab maybe qualified: {lab.get('name', 'Unknown')}")
                    else:
                        lab_dict['is_chemistry_lab'] = 'NO'
                        print(f"✗ Lab not qualified: {lab.get('name', 'Unknown')}")
                
                # Extract reasoning from line starting with "2."
                reason_line = next((line for line in lines if line.strip().startswith('2.')), '')
                if reason_line:
                    lab_dict['research_reason'] = reason_line.replace('2.', '').strip()
                else:
                    # Fallback: find first non-numbered line
                    reason_line = next((line for line in lines if line.strip() and not line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.'))), '')
                    lab_dict['research_reason'] = reason_line.strip()
                
                # Extract quotes from line starting with "3."
                quotes_line = next((line for line in lines if line.strip().startswith('3.')), '')
                if quotes_line:
                    lab_dict['research_quotes'] = quotes_line.replace('3.', '').strip()
                else:
                    lab_dict['research_quotes'] = ''
                
                # Extract test types from line starting with "4."
                tests_line = next((line for line in lines if line.strip().startswith('4.')), '')
                if tests_line:
                    tests_text = tests_line.replace('4.', '').strip()
                    # Remove "Types of tests offered by the lab:" prefix if present
                    if 'Types of tests offered by the lab:' in tests_text:
                        tests_text = tests_text.split('Types of tests offered by the lab:')[1].strip()
                    lab_dict['test_types'] = tests_text
                else:
                    lab_dict['test_types'] = ''
                
                # Extract industries from line starting with "5."
                industries_line = next((line for line in lines if line.strip().startswith('5.')), '')
                if industries_line:
                    industries_text = industries_line.replace('5.', '').strip()
                    # Remove "Industries the lab serves:" prefix if present
                    if 'Industries the lab serves:' in industries_text:
                        industries_text = industries_text.split('Industries the lab serves:')[1].strip()
                    lab_dict['industries_served'] = industries_text
                else:
                    lab_dict['industries_served'] = ''
                
                # Extract URL from line starting with "6."
                url_line = next((line for line in lines if line.strip().startswith('6.')), '')
                if url_line:
                    # Extract URL after "Homepage URL:" or "URL:"
                    url_text = url_line.replace('6.', '').strip()
                    if 'Homepage URL:' in url_text:
                        extracted_url = url_text.split('Homepage URL:')[1].strip()
                    elif 'URL:' in url_text:
                        extracted_url = url_text.split('URL:')[1].strip()
                    else:
                        # If no URL prefix found, check if it looks like a URL
                        if url_text.startswith(('http://', 'https://', 'www.')) or 'N/A' in url_text:
                            extracted_url = url_text
                        else:
                            extracted_url = ''
                    lab_dict['homepage_url'] = extracted_url
                else:
                    lab_dict['homepage_url'] = ''
                    
            except Exception as e:
                print(f"Warning: Error parsing research result: {e}")
                # Fallback parsing
                lab_dict['research_reason'] = research_result[:200] + '...' if research_result and len(research_result) > 200 else research_result
                lab_dict['research_quotes'] = ''
                lab_dict['test_types'] = ''
                lab_dict['industries_served'] = ''
                lab_dict['homepage_url'] = ''
                if research_result and "YES" in research_result.upper():
                    lab_dict['is_chemistry_lab'] = 'YES'
                elif research_result and "MAYBE" in research_result.upper():
                    lab_dict['is_chemistry_lab'] = 'MAYBE'
                else:
                    lab_dict['is_chemistry_lab'] = 'NO'
        
        lab_dict.pop('logo_url', None)
        results.append(lab_dict)
        
        # Write to JSON after each lab (append to existing results)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved lab {idx + 1}/{len(unprocessed_labs)} (ID: {lab_id}) to {output_file}")
    
    return results

def save_results(qualified_labs, output_filename):
    """Save qualified labs to output CSV file."""
    if not qualified_labs:
        print("No qualified labs found.")
        return
    
    df_output = pd.DataFrame(qualified_labs)
    df_output.to_csv(output_filename, index=False)
    print(f"Saved {len(qualified_labs)} qualified labs to {output_filename}")

def main(num_samples=20, seed=42, model_version="pro", input_file="input_labs.json", output_file="output_labs.json", prefilter_only=False):
    """Main function to orchestrate the lab filtering process."""
    
    # Load input JSON
    df = load_labs_json(input_file)
    if df is None:
        return
    
    # Test mode: randomly sample num_samples rows
    if num_samples > 0:
        print(f"Sampling {num_samples} random labs to process")
        df = df.sample(n=min(num_samples, len(df)), random_state=seed)
        df.reset_index(drop=True, inplace=True)
    else:
        # shuffle rows
        df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Using model: gemini-2.5-{model_version}")
    print(f"Output file: {output_file}")
    
    # Step 1: Apply pre-filters to reduce the dataset
    prefilter_file = output_file.replace('.json', '_prefiltered.json')
    print(f"\n=== STEP 1: Applying Pre-filters ===")
    print(f"Pre-filter results will be saved to: {prefilter_file}")
    
    prefilter_results = apply_pre_filters(df, prefilter_file, model_version)
    
    # Count pre-filter results
    passed_prefilters = [lab for lab in prefilter_results if lab.get('passes_prefilters', False)]
    failed_prefilters = [lab for lab in prefilter_results if not lab.get('passes_prefilters', False)]
    
    print(f"\n=== Pre-filter Results ===")
    print(f"Total labs processed: {len(prefilter_results)}")
    print(f"Passed pre-filters: {len(passed_prefilters)} ({len(passed_prefilters)/len(prefilter_results)*100:.1f}%)")
    print(f"Failed pre-filters: {len(failed_prefilters)} ({len(failed_prefilters)/len(prefilter_results)*100:.1f}%)")
    
    # Breakdown by filter type
    website_no = len([lab for lab in prefilter_results if lab.get('website_exists') == 'NO'])
    industry_no = len([lab for lab in prefilter_results if lab.get('chemistry_industry') == 'NO'])
    commercial_no = len([lab for lab in prefilter_results if lab.get('commercial_testing') == 'NO'])
    
    print(f"\nFilter breakdown:")
    print(f"  Website doesn't exist: {website_no} labs ({website_no/len(prefilter_results)*100:.1f}%)")
    print(f"  Non-chemistry industry: {industry_no} labs ({industry_no/len(prefilter_results)*100:.1f}%)")
    print(f"  Internal testing only: {commercial_no} labs ({commercial_no/len(prefilter_results)*100:.1f}%)")
    
    if prefilter_only:
        print(f"\nPre-filtering completed. Results saved to {prefilter_file}")
        return prefilter_results
    
    # Step 2: Apply chemistry lab filter to all labs from prefilter results
    print(f"\n=== STEP 2: Applying Chemistry Lab Filter ===")
    print(f"Processing all {len(prefilter_results)} labs (deep filtering only labs with all YES prefilters)")
    
    # Create DataFrame from all prefilter results for chemistry filtering
    df = pd.DataFrame(prefilter_results)
    
    # Apply chemistry lab filtering
    chemistry_results = filter_chemistry_labs(df, output_file, model_version)
    
    print(f"\n=== Final Results ===")
    print(f"Chemistry research completed. Results saved to {output_file}")
    chemistry_labs = [lab for lab in chemistry_results if lab.get('is_chemistry_lab') == 'YES']
    maybe_labs = [lab for lab in chemistry_results if lab.get('is_chemistry_lab') == 'MAYBE']
    
    print(f"Final qualified chemistry labs: {len(chemistry_labs)} out of {len(df)} original labs ({len(chemistry_labs)/len(df)*100:.1f}%)")
    print(f"Maybe chemistry labs: {len(maybe_labs)} out of {len(df)} original labs ({len(maybe_labs)/len(df)*100:.1f}%)")
    print(f"Total labs that passed all filters: {len(chemistry_labs) + len(maybe_labs)} out of {len(df)} original labs")
    
    return chemistry_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find analytical chemistry labs')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to process. Set to 0 to process all.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--version', choices=['pro', 'flash'], default='pro', 
                        help='Select Gemini model version: "pro" for gemini-2.5-pro or "flash" for gemini-2.5-flash')
    parser.add_argument('--input', default='input_labs.json',
                            help='Input JSON file name (default: input_labs.json)')
    parser.add_argument('--output', default='output_labs.json',
                        help='Output JSON file name (default: output_labs.json)')
    parser.add_argument('--prefilter-only', action='store_true',
                        help='Only run pre-filters (website exists, chemistry industry, commercial testing) without the final chemistry lab filter')
    
    args = parser.parse_args()
    main(num_samples=args.num_samples, seed=args.seed, model_version=args.version, input_file=args.input, output_file=args.output, prefilter_only=args.prefilter_only)
