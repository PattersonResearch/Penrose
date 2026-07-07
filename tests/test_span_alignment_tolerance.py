from penrose.pipeline.extract import span_in_text


def test_span_matches_rendered_markdown_emphasis():
    source = "When realized drift **exceeds** the funding-implied drift, returns improve."
    span = "When realized drift exceeds the funding-implied drift, returns improve."

    assert span_in_text(span, source)


def test_span_matches_unicode_minus_folded_to_ascii():
    source = "The signal is realized_return \u2212 funding_implied_drift over the next day."
    span = "realized_return - funding_implied_drift"

    assert span_in_text(span, source)


def test_span_matches_unicode_arrow_folded_to_ascii():
    source = "Regime transition: calm \u2192 stressed after funding shocks."
    span = "calm -> stressed"

    assert span_in_text(span, source)


def test_span_matches_em_dash_folded_to_hyphen():
    source = "Funding pressure rises\u2014and basis widens."
    span = "rises-and basis widens"

    assert span_in_text(span, source)


def test_span_preserves_whitespace_tolerance():
    source = "Funding pressure rises\n\nwhen basis widens."
    span = "Funding pressure   rises when\tbasis widens."

    assert span_in_text(span, source)


def test_span_accepts_non_contiguous_fully_verbatim_sentences():
    source = (
        "Entry day t+1 buys the breakout above the prior twenty day high. "
        "Position sizing is capped at one unit and fees are deducted daily. "
        "Exit closes the position at the close of day t+2."
    )
    span = (
        "Entry day t+1 buys the breakout above the prior twenty day high. "
        "Exit closes the position at the close of day t+2."
    )

    assert span_in_text(span, source)


def test_span_rejects_fabricated_words_and_numbers():
    source = "The strategy returned 3.1 percent during the sample."
    span = "The strategy returned 9.7 percent during the sample."

    assert not span_in_text(span, source)


def test_span_rejects_non_contiguous_span_with_fabricated_sentence():
    source = (
        "Entry day t+1 buys the breakout above the prior twenty day high. "
        "Position sizing is capped at one unit and fees are deducted daily. "
        "Exit closes the position at the close of day t+2."
    )
    span = (
        "Entry day t+1 buys the breakout above the prior twenty day high. "
        "The strategy earns a fabricated Sharpe of 9.9 after costs."
    )

    assert not span_in_text(span, source)


def test_short_fabricated_metric_fragment_does_not_fallback_match():
    source = "The reported Sharpe was 1.2 after transaction costs."

    assert not span_in_text("Sharpe of 9.9", source)


def test_empty_whitespace_and_marker_only_spans_never_match():
    source = "When realized drift **exceeds** the funding-implied drift."

    for span in ["", "   \n\t", "**", "__", "*", "`", " **  ` __ * ", "and then"]:
        assert not span_in_text(span, source)
