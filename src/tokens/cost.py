"""Cost estimation for API calls."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderPricing:
    """Pricing information for a provider."""

    provider: str
    input_price_per_1m: float  # Price per 1M input tokens
    output_price_per_1m: float  # Price per 1M output tokens
    notes: str = ""


# Approximate pricing (may vary - check provider docs for current rates)
PROVIDER_PRICING = {
    "cerebras": ProviderPricing(
        provider="cerebras",
        input_price_per_1m=0.10,
        output_price_per_1m=0.10,
        notes="Estimated - check Cerebras pricing",
    ),
    "groq": ProviderPricing(
        provider="groq",
        input_price_per_1m=0.05,
        output_price_per_1m=0.08,
        notes="Estimated - Groq has various tiers",
    ),
    "gemini": ProviderPricing(
        provider="gemini",
        input_price_per_1m=0.075,
        output_price_per_1m=0.30,
        notes="Gemini 2.5 Flash pricing",
    ),
    "cohere": ProviderPricing(
        provider="cohere",
        input_price_per_1m=0.50,
        output_price_per_1m=1.50,
        notes="Command R+ pricing",
    ),
}


@dataclass
class CostEstimate:
    """Cost estimate for an API call or aggregation."""

    provider: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        """Total cost in dollars."""
        return self.input_cost + self.output_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
        }


class CostCalculator:
    """Calculate costs for API usage."""

    def __init__(self, custom_pricing: dict[str, ProviderPricing] | None = None):
        """Initialize with optional custom pricing."""
        self.pricing = PROVIDER_PRICING.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)

    def estimate_cost(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEstimate:
        """Estimate cost for given token counts."""
        pricing = self.pricing.get(provider)

        if pricing is None:
            # Use default pricing if provider not found
            pricing = ProviderPricing(
                provider=provider,
                input_price_per_1m=0.10,
                output_price_per_1m=0.30,
            )

        input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_1m

        return CostEstimate(
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
        )

    def estimate_batch_cost(
        self,
        provider: str,
        calls: list[tuple[int, int]],  # List of (input_tokens, output_tokens)
    ) -> CostEstimate:
        """Estimate total cost for a batch of calls."""
        total_input = sum(c[0] for c in calls)
        total_output = sum(c[1] for c in calls)
        return self.estimate_cost(provider, total_input, total_output)

    def compare_format_costs(
        self,
        provider: str,
        base_input_tokens: int,
        format_overheads: dict[str, int],  # format -> additional output tokens
        base_output_tokens: int = 100,
    ) -> dict[str, CostEstimate]:
        """Compare costs across formats with different overheads."""
        results = {}

        for format_type, overhead in format_overheads.items():
            results[format_type] = self.estimate_cost(
                provider,
                base_input_tokens,
                base_output_tokens + overhead,
            )

        return results

    def get_pricing(self, provider: str) -> ProviderPricing | None:
        """Get pricing for a provider."""
        return self.pricing.get(provider)

    def set_pricing(self, provider: str, pricing: ProviderPricing) -> None:
        """Set or update pricing for a provider."""
        self.pricing[provider] = pricing
