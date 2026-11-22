from fos_triage.parse_search import parse_outcome, parse_result_count, parse_search_page

HTML = """
<html><body>
<p role="status">Your search returned 1,234 results</p>
<ul class="search-results">
  <li>
    <a href="decision/DRN-1000001.pdf" class="search-result">
      <h3>Decision Reference DRN-1000001</h3>
      <div class="search-result__info">
        <div class="search-result__info-main">
          <em>6 Oct 2024</em>
          <span class="search-result__separator"></span>
          Example Bank plc
          <span class="search-result__separator"></span>
          Not upheld
        </div>
        <span class="search-result__tag tag-type">Banking and Payments</span>
      </div>
      <div class="search-result__desc">
        The complaint Ms A complains about a delayed transfer.
        <strong>(3 pages)</strong>
      </div>
    </a>
  </li>
</ul>
</body></html>
"""


def test_count_and_row():
    assert parse_result_count(HTML) == 1234
    rows = parse_search_page(HTML)
    assert len(rows) == 1
    row = rows[0]
    assert row["reference"] == "DRN-1000001"
    assert row["date"] == "2024-10-06"
    assert row["business"] == "Example Bank plc"
    assert row["outcome"] == "not_upheld"
    assert row["upheld"] == 0
    assert row["product"] == "Banking and Payments"
    assert "delayed transfer" in row["snippet"]
    assert "pages" not in row["snippet"].lower()


def test_singular_result_count():
    assert parse_result_count("Your search returned 1 result") == 1
    assert parse_result_count("Your search returned 0 results") == 0


def test_outcome_order():
    assert parse_outcome("Not upheld") == ("not_upheld", 0)
    assert parse_outcome("Upheld") == ("upheld", 1)
