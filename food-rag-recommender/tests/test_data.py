from food_rag.data import build_document, build_metadata, normalize_food_item, unique_ids


def test_dataset_loads_and_is_normalised(foods):
    assert len(foods) >= 40
    for f in foods:
        assert isinstance(f["food_id"], str)
        assert isinstance(f["food_ingredients"], list)
        assert isinstance(f["food_calories_per_serving"], int)
        assert f["taste_profile"]


def test_missing_fields_get_defaults():
    item = normalize_food_item({"food_name": "Mystery Dish"}, 4)
    assert item["food_id"] == "5"
    assert item["cuisine_type"] == "Unknown"
    assert item["food_ingredients"] == []
    assert item["food_calories_per_serving"] == 0
    assert item["taste_profile"] == ""


def test_string_ingredients_are_split():
    item = normalize_food_item({"food_name": "X", "food_ingredients": "Rice, Beans ,Corn"}, 0)
    assert item["food_ingredients"] == ["Rice", "Beans", "Corn"]


def test_document_contains_key_fields(foods):
    doc = build_document(foods[0])
    for fragment in ("Name: Apple Pie", "Cuisine: American", "Cinnamon", "Nutrition:"):
        assert fragment in doc


def test_metadata_is_chroma_compatible(foods):
    for f in foods:
        for value in build_metadata(f).values():
            assert isinstance(value, (str, int, float, bool))
    meta = build_metadata(foods[0])
    assert meta["protein_g"] == 2.0 and meta["carbs_g"] == 42.0


def test_unique_ids_handles_duplicates():
    foods = [{"food_id": "1"}, {"food_id": "1"}, {"food_id": "2"}, {"food_id": "1"}]
    assert unique_ids(foods) == ["1", "1_1", "2", "1_2"]
