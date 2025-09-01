Data Source: 2025/08/06

added back the industry filter, and updated the industry filter to explicitly filter out Medical/clinical testing, Construction testing, and Biological/microbiological testing

```
=== Final Results ===
Chemistry research completed. Results saved to output_labs.json
Final qualified chemistry labs: 33 out of 200 original labs (16.5%)
Maybe chemistry labs: 1 out of 200 original labs (0.5%)
Total labs that passed all filters: 34 out of 200 original labs
Saved 33 YES-qualified labs to output_labs_yes_only.json
```

Deployment (as of 2025/09/01): http://13.57.32.207:3001

Prompt for industry filter:
```
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
    Determine if this lab primarily serves chemistry-related industries or non-chemistry industries.
    
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
    4. Percentage estimate of chemistry-related vs non-chemistry work (if determinable)
    
    ## Lab Information:
    {lab_info}
```


Run partially:
```sh
python3 find_chemistry_labs.py --num_samples 200 --start-stage 2
```
