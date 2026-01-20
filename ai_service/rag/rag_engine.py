import os

class RAGEngine:
    def __init__(self, data_path="rag/data/msme_policies.md", index_path="rag/faiss_index"):
        self.data_path = data_path
        self.index_path = index_path
        self.is_simulated = True  # Always use simulation mode for now
        self.qa_chain = None
        
        print(f"✓ RAG Engine initialized in SIMULATION MODE for {data_path}")

    def query(self, question):
        return self._simulated_query(question)

    def _simulated_query(self, question):
        """Simple keyword matching fallback for demo without API key."""
        question = question.lower()
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            sections = content.split('## ')
            best_match = None
            max_hits = 0

            keywords = question.split()
            for section in sections:
                if not section.strip(): continue
                hits = sum(1 for kw in keywords if len(kw) > 3 and kw in section.lower())
                if hits > max_hits:
                    max_hits = hits
                    best_match = section

            if best_match:
                return "SIMULATION MODE: " + best_match.strip()
            else:
                return "SIMULATION MODE: I couldn't find a specific match in my database for that. Try asking about a different category."
        except Exception as e:
            return f"Error in simulation mode: {str(e)}"

# Instances for different purposes
_policy_engine = None
_supplier_engine = None

def get_policy_engine():
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = RAGEngine(data_path="rag/data/msme_policies.md", index_path="rag/faiss_policy")
    return _policy_engine

def get_supplier_engine():
    global _supplier_engine
    if _supplier_engine is None:
        _supplier_engine = RAGEngine(data_path="rag/data/suppliers.md", index_path="rag/faiss_suppliers")
    return _supplier_engine
