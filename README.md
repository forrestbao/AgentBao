# AgentBao
AI agents built by Bao. 

## Agent 1: X-Chemists (Like X-Men)

Agent 1 is a cross-lingual customer discovery agent that discovers customers that work in biology, medicine, or chemistry. 

1. Given a keyword (say sodium chloride), it goes to PubMed and uses PubMed's API to find top 100 papers that match the keyword published with in the past 10 years. 
2. Then it extracts the author names, affiliations, email addresses, postal addresses, and phone numbers (if any) for each paper retrieved. 
3. With the info of authors in English, the agent shall find the authors in the Chinese-speaking world. Below are some strategies but not limited to. The LLM should also figure out ways to combine strategies. 
   * To get the authors' affiliations in Chinese, try two approaches: translate and then Google, or directly Google but limit the search to Chinese info sources. 
   * Email addresses can be used as is in web search to locate the author but limit the search to Chinese info sources. 
   * Author names can be translated into possible Chinese names -- this is non-deterministic because different Chinese names can be transliterated or Romanized to the same English name. Try common Chinese names, especially last/family names. 
4. Finally, the agent should print out the English and Chinese info of authors in two columns. 

## Agent 2: Find-me-a-lab

TBD. 