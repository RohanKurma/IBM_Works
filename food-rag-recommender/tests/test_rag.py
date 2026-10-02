from food_rag.rag import FoodRAG, fallback_response, prepare_context, simple_comparison
from food_rag.render import render_results
from food_rag.search import semantic_search


class StubLLM:
    name = "stub"

    def __init__(self, reply="Here is a lovely recommendation that is definitely longer than fifty characters."):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


class BrokenLLM:
    name = "broken"

    def generate(self, prompt):
        raise RuntimeError("API down")


def test_context_lists_top_three(collection):
    results = semantic_search(collection, "pasta", 5)
    context = prepare_context(results)
    assert "Option 3:" in context and "Option 4:" not in context
    assert "Key ingredients:" in context


def test_recommend_uses_llm_and_grounds_prompt(collection, cuisines):
    llm = StubLLM()
    answer = FoodRAG(collection, llm, cuisines).recommend("Italian food under 500 calories")
    assert answer.used_llm and answer.answer == llm.reply
    assert all(r["cuisine_type"] == "Italian" for r in answer.results)
    assert answer.results[0]["food_name"] in llm.prompts[0]


def test_recommend_falls_back_when_llm_fails(collection, cuisines):
    answer = FoodRAG(collection, BrokenLLM(), cuisines).recommend("spicy noodles")
    assert not answer.used_llm
    assert answer.results[0]["food_name"] in answer.answer


def test_recommend_without_llm(collection, cuisines):
    rag = FoodRAG(collection, None, cuisines)
    assert "template" in rag.llm_name
    assert rag.recommend("healthy breakfast").answer


def test_impossible_filters_are_relaxed(collection, cuisines):
    answer = FoodRAG(collection, None, cuisines).recommend("Dessert under 10 calories")
    assert answer.results and answer.filters.is_empty()


def test_history_is_included_in_prompt(collection, cuisines):
    llm = StubLLM()
    FoodRAG(collection, llm, cuisines).recommend("something lighter", [("pizza ideas", "Try Margherita")])
    assert "pizza ideas" in llm.prompts[0]


def test_compare_and_fallbacks(collection, cuisines):
    text, r1, r2 = FoodRAG(collection, None, cuisines).compare("chocolate dessert", "healthy breakfast")
    assert r1 and r2 and "lighter" in text
    assert simple_comparison("a", "b", [], []) == "No results found for either query."
    assert "couldn't find" in fallback_response("x", [])


def test_render_escapes_html(collection):
    results = semantic_search(collection, "pie", 1)
    results[0]["food_name"] = "<script>alert(1)</script>"
    html = render_results(results)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "empty" in render_results([])
