"""
Week 1 Solutions - LLM Engineering Course
Challenges from Day 1 and Day 2
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from typing import Optional


# ============================================================================
# DAY 1 CHALLENGE: Website Summarizer with LLM
# ============================================================================

class WebsiteSummarizer:
    """
    Summarizes website content using an LLM.
    Can use OpenAI, Ollama, or any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        model: str = "llama3.2:1b",
    ):
        """
        Initialize the summarizer with a specific LLM endpoint.

        Args:
            base_url: Base URL for the LLM endpoint
            api_key: API key for authentication
            model: Model name to use
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""
        self.user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""

    def fetch_website_contents(self, url: str) -> str:
        """Fetch raw HTML content from a URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            # Simple text extraction from HTML
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []

                def handle_data(self, data):
                    self.text.append(data.strip())

            parser = TextExtractor()
            parser.feed(response.text)
            return " ".join(self.text)
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"

    def summarize(self, url: str) -> str:
        """Summarize a website by URL."""
        website_content = self.fetch_website_contents(url)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_prefix + website_content},
        ]

        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )

        return response.choices[0].message.content


# ============================================================================
# DAY 2 CHALLENGE: Email Subject Line Suggester
# ============================================================================

class EmailSubjectSuggester:
    """
    Suggests appropriate subject lines for email content.
    Commercial use case: built-in feature for email tools.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        model: str = "llama3.2:1b",
    ):
        """Initialize the email subject suggester."""
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = """
You are an expert email assistant that suggests concise, professional subject lines.
Generate 3-5 subject line options for the given email content.
Make them clear, concise, and relevant.
Format each suggestion on a new line with a bullet point.
"""

    def suggest_subject_line(self, email_body: str) -> str:
        """
        Suggest subject lines for email content.

        Args:
            email_body: The body of the email

        Returns:
            Suggested subject lines
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Please suggest subject lines for this email:\n\n{email_body}",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )

        return response.choices[0].message.content


# ============================================================================
# DAY 2 CHALLENGE: Multi-Provider LLM Interface
# ============================================================================

class MultiProviderLLM:
    """
    Unified interface to call different LLM providers using OpenAI-compatible endpoints.
    Supports: OpenAI, Ollama, Google Gemini, and any OpenAI-compatible endpoint.
    """

    PROVIDERS = {
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "default_model": "llama3.2:1b",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "default_model": "gemini-2.5-flash-lite",
        },
    }

    def __init__(self, provider: str = "ollama", model: Optional[str] = None):
        """
        Initialize the multi-provider LLM.

        Args:
            provider: One of 'ollama', 'openai', 'gemini'
            model: Optional model override
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        config = self.PROVIDERS[provider].copy()
        model = model or config["default_model"]

        # Load API key from environment if needed
        if provider == "openai":
            load_dotenv(override=True)
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "gemini":
            load_dotenv(override=True)
            api_key = os.getenv("GOOGLE_API_KEY")
        else:
            api_key = config.get("api_key", "ollama")

        self.client = OpenAI(base_url=config["base_url"], api_key=api_key)
        self.model = model
        self.provider = provider

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response from the LLM.

        Args:
            system_prompt: System instructions for the model
            user_prompt: The user's input

        Returns:
            Generated text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )

        return response.choices[0].message.content

    def compare_providers(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Compare responses from different providers (for testing).

        Args:
            system_prompt: System instructions
            user_prompt: The user's input

        Returns:
            Dictionary with responses from each provider
        """
        results = {}
        for provider_name in ["ollama"]:  # Can add more if keys are available
            try:
                llm = MultiProviderLLM(provider=provider_name)
                results[provider_name] = llm.generate(system_prompt, user_prompt)
            except Exception as e:
                results[provider_name] = f"Error: {str(e)}"

        return results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_day1_website_summarizer():
    """Example: Summarize a website"""
    print("=" * 60)
    print("DAY 1 CHALLENGE: Website Summarizer")
    print("=" * 60)

    summarizer = WebsiteSummarizer()

    urls = [
        "https://www.bbc.com",
        "https://www.wikipedia.org",
    ]

    for url in urls:
        try:
            print(f"\nSummarizing: {url}")
            print("-" * 40)
            summary = summarizer.summarize(url)
            print(summary)
        except Exception as e:
            print(f"Error: {e}")


def example_day2_email_subject():
    """Example: Generate email subject lines"""
    print("\n" + "=" * 60)
    print("DAY 2 CHALLENGE: Email Subject Line Suggester")
    print("=" * 60)

    suggester = EmailSubjectSuggester()

    sample_email = """
Hi team,

I wanted to follow up on the quarterly review meeting we had last week.
The feedback was incredibly valuable, and I've already started implementing
some of the suggestions.

Specifically, I've begun reorganizing the project timeline to account for
the extended testing phase, and I've scheduled meetings with the design team
to refine the UI mockups.

I'll have a full update ready for next week's standup.

Thanks again for all the thoughtful input!

Best regards,
Alex
"""

    print("\nEmail Content:")
    print(sample_email)
    print("\nSuggested Subject Lines:")
    print("-" * 40)
    suggestions = suggester.suggest_subject_line(sample_email)
    print(suggestions)


def example_day2_multi_provider():
    """Example: Using multiple LLM providers"""
    print("\n" + "=" * 60)
    print("DAY 2 CHALLENGE: Multi-Provider LLM")
    print("=" * 60)

    system_prompt = "You are a helpful assistant. Keep responses brief."
    user_prompt = "What are 3 interesting facts about AI?"

    # Using Ollama locally
    print("\nUsing Ollama (Local):")
    print("-" * 40)
    try:
        llm = MultiProviderLLM(provider="ollama", model="llama3.2:1b")
        response = llm.generate(system_prompt, user_prompt)
        print(response)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run examples
    example_day1_website_summarizer()
    example_day2_email_subject()
    example_day2_multi_provider()
