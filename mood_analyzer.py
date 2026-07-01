# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re

from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Emojis and slang carry strong sentiment on their own, so we treat
        # them as high-weight positive/negative words. preprocess() already
        # emits emojis (":)", "😂", "💀") and slang as their own tokens.
        positive_signals = {
            # Text emojis
            ":)", ":-)", ":d", ":')", "(:",
            # Unicode emojis
            "😂", "🥰", "😍", "❤", "❤️", "🔥", "🙌", "✨",
            # Slang
            "lol", "lmao", "goated", "slay", "vibes", "based",
        }
        negative_signals = {
            # Text emojis
            ":(", ":-(", ":'(", "):", ":/",
            # Unicode emojis
            "💀", "😭", "😡", "🙄", "😞", "😤",
            # Slang
            "mid", "cringe", "sus", "trash", "smh", "yikes",
        }

        # Fold the strong signals into the regular sentiment sets so they flow
        # through the same scoring + negation logic as ordinary words.
        self.positive_words = set(w.lower() for w in positive_words) | positive_signals
        self.negative_words = set(w.lower() for w in negative_words) | negative_signals

        # Intensity weights for words with a stronger connotation.
        # Any word not listed here uses DEFAULT_WEIGHT (a normal signal),
        # so "hate" (3) counts more than "dislike"/"bad" (1).
        self.word_weights: Dict[str, int] = {
            # Strong positive
            "love": 3,
            "amazing": 3,
            "awesome": 3,
            "excited": 2,
            "great": 2,
            # Strong negative
            "hate": 3,
            "terrible": 3,
            "awful": 3,
            "angry": 2,
            "stressed": 2,
        }

        # Emojis and slang are strong signals: give every one of them a high
        # weight without having to list each individually above.
        for signal in positive_signals | negative_signals:
            self.word_weights.setdefault(signal, 3)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")

        This version splits the text so that words, text emojis (":)",
        ":-(", ":D"), and any remaining single punctuation/emoji character
        each become their own token. That way "great!" -> ["great", "!"]
        and "happy :)" -> ["happy", ":)"] instead of gluing the symbol onto
        the word.
        """
        cleaned = text.strip().lower()

        # Order matters: match multi-character text emojis first so they are
        # not broken apart, then words, then any leftover single non-space
        # character (punctuation or a unicode emoji).
        token_pattern = re.compile(
            r":-?[()doOpP/\\|]"   # text emojis like :), :-(, :D, :/
            r"|[a-z0-9]+"          # words / numbers
            r"|[^\sa-z0-9]"        # any other single symbol (punctuation, emoji)
        )

        tokens = token_pattern.findall(cleaned)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    #: Weight used for any sentiment word not listed in self.word_weights.
    DEFAULT_WEIGHT = 1

    def word_weight(self, token: str) -> int:
        """Return the intensity weight for a sentiment word."""
        return self.word_weights.get(token, self.DEFAULT_WEIGHT)

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        positive_total, negative_total = self.score_components(text)
        return positive_total - negative_total

    def score_components(self, text: str) -> Tuple[int, int]:
        """
        Break the text into its total positive and negative contributions.

        Returns a ``(positive_total, negative_total)`` pair, both as
        non-negative magnitudes. This lets callers see whether a message has
        BOTH kinds of sentiment (mixed) instead of only the net score.

        Negation is applied here, so a negated positive word (e.g. "not
        happy") counts toward ``negative_total`` and vice versa.
        """
        tokens = self.preprocess(text)

        # Words that flip the meaning of the word that follows them.
        negation_words = {"not", "no", "never", "n't", "cannot", "cant"}

        positive_total = 0
        negative_total = 0
        negate = False  # True when the previous token was a negation word.

        for token in tokens:
            if token in negation_words:
                # Turn on negation for the NEXT sentiment word and move on.
                negate = True
                continue

            # Base polarity: +1 for positive, -1 for negative, 0 otherwise.
            if token in self.positive_words:
                polarity = 1
            elif token in self.negative_words:
                polarity = -1
            else:
                polarity = 0

            if polarity != 0:
                # Stronger words count for more (e.g. "hate" > "dislike").
                # Every occurrence is scored, so repeated words add up.
                weight = self.word_weight(token)
                if negate:
                    polarity = -polarity  # "not happy" -> negative

                if polarity > 0:
                    positive_total += weight
                else:
                    negative_total += weight

            # Negation only applies to the single word right after it,
            # so reset once we've processed a real token.
            negate = False

        return positive_total, negative_total

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the sentiment of a piece of text into a mood label.

        The mapping is:
          - has BOTH positive and negative parts -> "mixed"
          - otherwise net score > 0              -> "positive"
          - otherwise net score < 0              -> "negative"
          - otherwise (no sentiment at all)      -> "neutral"

        "mixed" is checked first so a message like "relaxed and stressed"
        (which nets to 0) is labeled "mixed" rather than "neutral".
        """
        positive_total, negative_total = self.score_components(text)

        if positive_total > 0 and negative_total > 0:
            return "mixed"
        if positive_total > negative_total:
            return "positive"
        if negative_total > positive_total:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
