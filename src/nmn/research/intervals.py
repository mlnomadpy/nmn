"""Closed rational intervals; every arithmetic endpoint is exact."""

from dataclasses import dataclass
from fractions import Fraction


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Fraction)):
        raise ValueError("expected a finite number or rational string")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("expected a finite rational value") from exc


@dataclass(frozen=True)
class RationalInterval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self):
        object.__setattr__(self, "lower", rational(self.lower))
        object.__setattr__(self, "upper", rational(self.upper))
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")

    @classmethod
    def point(cls, value):
        return cls(value, value)

    def __add__(self, other):
        other = other if isinstance(other, RationalInterval) else self.point(other)
        return RationalInterval(self.lower + other.lower, self.upper + other.upper)

    __radd__ = __add__

    def __neg__(self):
        return RationalInterval(-self.upper, -self.lower)

    def __sub__(self, other):
        return self + -(
            other if isinstance(other, RationalInterval) else self.point(other)
        )

    def __mul__(self, other):
        other = other if isinstance(other, RationalInterval) else self.point(other)
        products = [
            a * b for a in (self.lower, self.upper) for b in (other.lower, other.upper)
        ]
        return RationalInterval(min(products), max(products))

    __rmul__ = __mul__

    def square(self):
        a, b = self.lower * self.lower, self.upper * self.upper
        return RationalInterval(
            Fraction(0) if self.lower <= 0 <= self.upper else min(a, b), max(a, b)
        )

    def positive_reciprocal(self):
        if self.lower <= 0:
            raise ValueError("reciprocal requires a strictly positive interval")
        return RationalInterval(1 / self.upper, 1 / self.lower)

    def to_list(self):
        return [str(self.lower), str(self.upper)]
