import sys
sys.path.append("../src")
from preprocess import clean_text, build_dataset

def test_clean_text_removes_urls():
    """Test that URLs are removed from text."""
    text = "Check this out http://reddit.com it is great"
    cleaned = clean_text(text)
    assert "http" not in cleaned, "URL was not removed!"
    print("test_clean_text_removes_urls passed!")

def test_clean_text_lowercases():
    """Test that text is lowercased."""
    text = "I Am Feeling SAD Today"
    cleaned = clean_text(text)
    assert cleaned == cleaned.lower(), "Text was not lowercased!"
    print("✅ test_clean_text_lowercases passed!")

def test_dataset_splits():
    """Test that dataset splits are correct sizes."""
    train, val, test, le = build_dataset()
    assert len(train) == 630, f"Expected 630 train samples, got {len(train)}"
    assert len(val) == 135, f"Expected 135 val samples, got {len(val)}"
    assert len(test) == 135, f"Expected 135 test samples, got {len(test)}"
    print("test_dataset_splits passed!")

def test_label_classes():
    """Test that all 3 classes are present."""
    train, val, test, le = build_dataset()
    assert set(le.classes_) == {"anxiety", "depression", "neutral"}
    print("test_label_classes passed!")

if __name__ == "__main__":
    test_clean_text_removes_urls()
    test_clean_text_lowercases()
    test_dataset_splits()
    test_label_classes()
    print("\n All tests passed!")
