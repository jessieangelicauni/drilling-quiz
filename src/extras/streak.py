from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class StreakState:
    count: int = 0
    last_completed: date | None = None

    def record_completion(self, today: date) -> "StreakState":
        if self.last_completed == today:
            return self
        if self.last_completed == today - timedelta(days=1):
            return StreakState(count=self.count + 1, last_completed=today)
        return StreakState(count=1, last_completed=today)

    def is_broken(self, today: date) -> bool:
        return self.last_completed is not None and self.last_completed < today - timedelta(days=1)

    def to_dict(self) -> dict:
        return {"count": self.count,
                "last_completed": self.last_completed.isoformat() if self.last_completed else None}

    @staticmethod
    def from_dict(d: dict) -> "StreakState":
        lc = date.fromisoformat(d["last_completed"]) if d["last_completed"] else None
        return StreakState(count=d["count"], last_completed=lc)


if __name__ == "__main__":
    d0 = date(2026, 10, 1)
    s = StreakState()
    s = s.record_completion(d0)
    assert s.count == 1
    s = s.record_completion(d0 + timedelta(days=1))
    assert s.count == 2
    s = s.record_completion(d0 + timedelta(days=1))
    assert s.count == 2
    s = s.record_completion(d0 + timedelta(days=3))
    assert s.count == 1

    broken = StreakState(count=5, last_completed=d0)
    assert broken.is_broken(d0 + timedelta(days=2))
    assert not broken.is_broken(d0 + timedelta(days=1))

    d = s.to_dict()
    assert StreakState.from_dict(d).count == s.count

    print("all smoke tests passed")
