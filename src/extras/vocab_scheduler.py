from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import fsrs

_SCHEDULER = fsrs.Scheduler()


def new_vocab_card(card_id: int) -> fsrs.Card:
    return fsrs.Card(card_id=card_id)


def review(card: fsrs.Card, correct: bool,
           now: datetime | None = None) -> tuple[fsrs.Card, fsrs.ReviewLog]:
    rating = fsrs.Rating.Good if correct else fsrs.Rating.Again
    return _SCHEDULER.review_card(card, rating, review_datetime=now)


def is_due(card: fsrs.Card, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    return card.due <= now


def due_vocab_cards(cards: dict[str, fsrs.Card],
                     now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    due = [(item_id, card) for item_id, card in cards.items() if card.due <= now]
    due.sort(key=lambda pair: pair[1].due)
    return [item_id for item_id, _ in due]


if __name__ == "__main__":
    from datetime import timedelta

    card = new_vocab_card(1)
    assert is_due(card)

    card, _ = review(card, correct=True)
    assert not is_due(card)

    far_future = card.due + timedelta(days=1)
    assert is_due(card, now=far_future)

    cards = {"a": new_vocab_card(1), "b": new_vocab_card(2)}
    cards["b"], _ = review(cards["b"], correct=True)
    assert due_vocab_cards(cards) == ["a"]

    print("all smoke tests passed")
