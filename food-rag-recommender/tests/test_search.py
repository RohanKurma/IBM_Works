from food_rag.search import SearchFilters, build_where_clause, parse_query_filters, semantic_search


def test_where_clause_none_without_filters():
    assert build_where_clause(SearchFilters()) is None


def test_where_clause_single_filter_is_not_wrapped():
    assert build_where_clause(SearchFilters(cuisines=["Thai"])) == {"cuisine_type": "Thai"}


def test_where_clause_combines_with_and():
    where = build_where_clause(SearchFilters(cuisines=["Thai", "Indian"], max_calories=400, min_protein_g=10))
    assert where == {"$and": [
        {"cuisine_type": {"$in": ["Thai", "Indian"]}},
        {"calories": {"$lte": 400}},
        {"protein_g": {"$gte": 10.0}},
    ]}


def test_basic_search_returns_relevant_results(collection):
    results = semantic_search(collection, "chocolate dessert", 3)
    assert len(results) == 3
    assert any("Chocolate" in r["food_name"] for r in results)
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_filters_are_respected(collection):
    filters = SearchFilters(cuisines=["Italian"], max_calories=300)
    results = semantic_search(collection, "fresh", 5, filters)
    assert results
    assert all(r["cuisine_type"] == "Italian" and r["food_calories_per_serving"] <= 300 for r in results)


def test_ingredient_post_filter_overfetches(collection):
    results = semantic_search(collection, "dinner", 3, SearchFilters(must_include_ingredient="chicken"))
    assert results and all(any("chicken" in i.lower() for i in r["food_ingredients"]) for r in results)
    assert collection.last_query["n_results"] == 15


def test_n_results_capped_at_collection_size(collection):
    semantic_search(collection, "food", 500)
    assert collection.last_query["n_results"] == collection.count()


def test_parse_query_filters(cuisines):
    f = parse_query_filters("What Italian dishes do you recommend under 400 calories?", cuisines)
    assert f.cuisines == ["Italian"] and f.max_calories == 400

    f = parse_query_filters("something spicy, max 300 kcal", cuisines)
    assert f.max_calories == 300 and f.cuisines == []

    assert parse_query_filters("comfort food for a cold evening", cuisines).is_empty()
