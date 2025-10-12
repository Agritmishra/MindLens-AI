# insight_engine.py
"""
InsightEngine — small, robust analysis engine for MindLens.
Loads (if available):
 - a fine-tuned text-classifier (recommended)
 - or zero-shot classifiers (fallback)
 - a summarization pipeline (optional)
If any model load fails we gracefully fallback to lightweight heuristics
so the app continues running.
"""

import re
import random
from collections import Counter

# try importing transformers pipelines; if not available we'll continue with heuristics
try:
    from transformers import pipeline
except Exception:
    pipeline = None

random.seed(42)


class InsightEngine:
    def __init__(
        self,
        classifier_model: str | None = None,
        summarizer_model: str | None = None,
    ):
        # user-provided model IDs (set these when available)
        self.classifier_model = classifier_model  # e.g. "youruser/emotion-tiny-distilbert"
        self.summarizer_model = summarizer_model  # e.g. "sshleifer/distilbart-cnn-12-6"

        # runtime handles
        self.classifier = None
        self.zero_shot = None
        self.summarizer = None

        # candidate zero-shot models (light -> heavier)
        self.zero_shot_candidates = [
            "typeform/distilbert-base-uncased-mnli",
            "valhalla/distilbart-mnli-12-1",
        ]

        # try to load classifier (text-classification) first
        if pipeline is not None:
            self._try_load_models()

        # small prompt banks and micro-actions (human wording)
        self._build_prompt_banks()

        # simple lexicons
        self._positive_words = set(
            ["good", "great", "happy", "proud", "ok", "well", "better", "calm", "relieved", "optimistic"])
        self._negative_words = set(
            ["sad", "depressed", "angry", "stressed", "anxious", "worried", "scared", "upset", "tired", "down"])

    def _try_load_models(self):
        # 1) try user classifier (fine-tuned)
        if self.classifier_model:
            try:
                # prefer text-classification for fine-tuned emotion classifier
                self.classifier = pipeline("text-classification", model=self.classifier_model, device=-1)
            except Exception:
                self.classifier = None

        # 2) try zero-shot pipeline (if no fine-tuned)
        if not self.classifier and pipeline is not None:
            for m in self.zero_shot_candidates:
                try:
                    self.zero_shot = pipeline("zero-shot-classification", model=m, device=-1)
                    break
                except Exception:
                    self.zero_shot = None

        # 3) attempt summarizer (optional)
        if self.summarizer_model and pipeline is not None:
            try:
                self.summarizer = pipeline("summarization", model=self.summarizer_model, device=-1)
            except Exception:
                self.summarizer = None

    def _build_prompt_banks(self):
        self.mood_prompts = {
            "optimistic": [
                "What small progress are you proud of today?",
                "What makes you believe tomorrow can be even better?",
                "What would your best self say to you right now?",
                "What challenge have you already turned into an opportunity?",
                "How can you use this positive energy to help someone else?",
                "Which dream feels most alive inside you today?",
                "What gives your motivation meaning right now?",
                "What made you smile unexpectedly this week?",
                "If things go right, what will you thank yourself for later?",
                "What story would you tell yourself to keep going?",
                "What’s one daily ritual that keeps your optimism alive?",
                "What future version of yourself are you becoming?",
                "What’s something today that quietly worked out well?",
                "How can you channel your excitement into focus?",
                "What success from your past can you use for momentum?",
                "If you were mentoring someone, what lesson would you share?",
                "What has growth taught you about patience?",
                "Which parts of you are evolving in silence?",
                "What would a fearless step look like today?",
                "How can you turn hope into practical effort?"
            ],
            "sad": [
                "What do you wish someone could remind you right now?",
                "What is one gentle truth you can accept without resistance?",
                "Who or what gives you comfort when words don’t work?",
                "What could healing look like today — even in a small way?",
                "What emotion sits just beneath your sadness?",
                "What’s something or someone you miss that gave you warmth?",
                "If your sadness had a color, what would it be — and why?",
                "What would it mean to forgive yourself for feeling low?",
                "What helped you recover before — could it help again?",
                "What would compassion for yourself look like in this moment?",
                "What’s one thing that still matters despite the pain?",
                "How might you transform this sadness into understanding?",
                "Who do you feel safe enough to share your truth with?",
                "What memory still carries warmth despite sadness?",
                "When was the last time you surprised yourself with resilience?",
                "What would you like to hear from a loved one right now?",
                "If sadness had wisdom, what would it teach you?",
                "What are you still holding on to that needs release?",
                "What could self-kindness mean for you today?",
                "What would healing slowly, on your own terms, look like?"
            ],
            "anxious": [
                "What is one fear that doesn’t deserve your attention today?",
                "What would you tell a friend who felt exactly this way?",
                "Which part of your worry is based on imagination, not fact?",
                "If calmness had a texture, how would it feel right now?",
                "What outcome are you trying too hard to control?",
                "What would you do if you weren’t afraid of failure?",
                "What’s the smallest, real action you can take next?",
                "How can you ground yourself through your senses right now?",
                "What would 5-minute peace look like for you today?",
                "Which past anxiety turned out easier than expected?",
                "What can you accept instead of trying to fix immediately?",
                "What’s a truth that can calm your nervous system?",
                "What pattern of thought keeps repeating lately?",
                "If you zoomed out, what would this worry look like from afar?",
                "What’s something that went right recently despite fear?",
                "When did you last handle uncertainty successfully?",
                "What might help you feel 10% more at ease?",
                "What can you delegate, delay, or drop right now?",
                "If you were calm already, what would you do next?",
                "What does your body need most right now — rest, air, or stillness?"
            ],
            "angry": [
                "What part of you is really asking to be understood?",
                "Which value or boundary feels violated right now?",
                "What does this anger want you to protect?",
                "If you stepped back, what truth would remain under the heat?",
                "How could you express this anger safely and constructively?",
                "What might forgiveness look like — not for them, but for you?",
                "What outcome do you actually want beyond being ‘right’?",
                "What part of this conflict isn’t worth your energy?",
                "What could you learn about yourself from this anger?",
                "What pattern keeps repeating when you feel triggered?",
                "What need or fear hides behind the irritation?",
                "How can you respond instead of react next time?",
                "What is this anger trying to tell you about your values?",
                "If your anger had a voice, what would it say gently?",
                "What boundary could you communicate more clearly next time?",
                "When was the last time anger helped you make change?",
                "How can you transform this energy into calm power?",
                "What would resolution look like, even imperfectly?",
                "How might patience protect your peace here?",
                "What truth needs saying — without aggression, just clarity?"
            ],
            "confused": [
                "What’s one question you could answer that would make things clearer?",
                "What are you afraid might happen if you choose wrong?",
                "If you knew you couldn’t fail, what would you do?",
                "Which option feels lighter or more natural in your body?",
                "What are you over-thinking that might need feeling instead?",
                "What would clarity look like — describe it in one sentence.",
                "If a wiser version of you existed, what advice would it give?",
                "What’s the one fact you’re absolutely sure of right now?",
                "Which direction feels just 10% more right than the other?",
                "How can you make a small experiment instead of a big decision?",
                "What’s one step you can take that doesn’t close any doors?",
                "What have you learned in past moments of uncertainty?",
                "What outcome matters most — not what looks best?",
                "What could you test instead of analyze endlessly?",
                "Which question are you avoiding because it matters most?",
                "What information are you waiting for that you could find today?",
                "If you trusted yourself, what would you pick?",
                "Who might help you think more clearly about this?",
                "What happens if you do nothing for 24 hours?",
                "What decision will future you thank you for?"
            ],
            "neutral": [
                "What could make today 1% more interesting?",
                "What moment of balance can you appreciate right now?",
                "How can you turn this calm into quiet growth?",
                "What feels steady and reliable in your life today?",
                "What’s one new curiosity you could explore gently?",
                "What habit has quietly made your life better?",
                "What might be quietly blooming in this neutral moment?",
                "If nothing feels urgent, what’s quietly calling your name?",
                "What simple joy could you add to your day right now?",
                "How could you celebrate ordinary peace?",
                "What small kindness could you offer someone today?",
                "How can you use this balance to reset your goals?",
                "What small improvement could you make without effort?",
                "What’s quietly teaching you patience right now?",
                "What routine deserves appreciation, not automation?",
                "What are you taking for granted that’s actually working?",
                "What soft moment recently went unnoticed but mattered?",
                "How can you turn ‘ordinary’ into ‘meaningful’ today?",
                "If balance were fragile, how would you protect it?",
                "What do you feel grateful for that doesn’t need words?"
            ],
            "motivated": [
                "What small win deserves celebration right now?",
                "How will today’s effort benefit your future self?",
                "What’s the next micro-goal that moves you forward?",
                "What reminds you why you started this journey?",
                "How can you make consistency easier than motivation?",
                "What does momentum feel like in your body?",
                "What’s one distraction you can cut for 24 hours?",
                "Who’s someone you’d like to make proud through this work?",
                "What’s your minimum-action version of progress today?",
                "What tiny win could make you smile at day’s end?",
                "If you do nothing else today, what one task matters most?",
                "What would you tell yourself at the finish line?",
                "Which habits feel energizing instead of exhausting?",
                "What would working smarter (not harder) look like?",
                "How can you reward yourself meaningfully tonight?",
                "What’s one task you’ve been postponing that needs 5 minutes?",
                "How can you simplify your next step right now?",
                "Who could you collaborate with for inspiration?",
                "How would a disciplined version of you spend this hour?",
                "What small progress deserves to be documented?"
            ],
            "calm": [
                "What is working right now that you’d like to protect?",
                "What brings this sense of calm — and how can you extend it?",
                "Which thoughts disturb your peace — can you let one pass?",
                "What slow ritual restores your inner balance?",
                "What boundary keeps this serenity intact?",
                "Who or what supports your peace quietly?",
                "What moment of stillness can you revisit later today?",
                "What’s one thing you could do slower, more mindfully?",
                "How could you share calm with someone else today?",
                "What sound or scent helps you return to center?",
                "How does calm feel physically — can you notice it now?",
                "How will you end today gently?",
                "What do you want to protect your energy from this week?",
                "What does contentment mean for you right now?",
                "What would happen if you allowed yourself full rest?",
                "Which parts of your routine nurture your peace best?",
                "What scenery makes you feel grounded instantly?",
                "How can you respond more softly to the next challenge?",
                "What reminder helps you maintain balance under stress?",
                "How could you be even kinder to your mind today?"
            ]
        }

        self.micro_actions = {
            "optimistic": [
                "Write down 3 moments that went right today.",
                "Do something kind for a stranger or friend.",
                "Plan one realistic, exciting goal for next week.",
                "Create a short gratitude list and read it aloud.",
                "Send an encouraging message to someone who needs it.",
                "Stretch for 2 minutes while visualizing success.",
                "Write an appreciation message to yourself.",
                "Make a 10-minute plan for tomorrow with enthusiasm.",
                "Drink water mindfully and breathe with intention.",
                "Note one quality that makes you resilient.",
                "Read something uplifting for 5 minutes.",
                "Smile intentionally at someone today."
            ],
            "sad": [
                "Go for a 5-minute walk and notice colors or textures.",
                "Listen to one comforting song fully with eyes closed.",
                "Write one line about what you wish to release.",
                "Message a friend without pretending you’re fine.",
                "Drink water and rest your eyes for 2 minutes.",
                "Write a small letter to your past self — forgive them.",
                "Hold something warm — tea, a pet, or a blanket — consciously.",
                "Note three small things still in your control.",
                "Write a list titled ‘Things that will not last forever’.",
                "Journal one lesson sadness has taught you before.",
                "Take a warm shower and imagine washing away heaviness.",
                "Look at the sky for one full minute."
            ],
            "anxious": [
                "Practice box-breathing (4-4-4-4) for one minute.",
                "Name 5 things you see, 4 you feel, 3 you hear.",
                "Write down your top worry, then one thing you can do about it.",
                "Do a 10-minute body scan from head to toe.",
                "Stretch your neck and shoulders slowly.",
                "Repeat the phrase: ‘I’m safe in this moment.’",
                "Lower screen brightness and sit in silence for 2 minutes.",
                "Write one reassuring truth from your past experience.",
                "Identify the fear that has not come true — yet you survived.",
                "Move your body — 10 slow breaths while walking.",
                "Look around and name 3 stable things in your environment.",
                "Write down your breathing pattern — notice it change."
            ],
            "angry": [
                "Leave the room for 30 seconds before reacting.",
                "Exhale longer than you inhale — 3 times.",
                "Journal freely for 3 minutes about what’s under your anger.",
                "Write the word ‘Pause’ and look at it until tension drops.",
                "Squeeze a stress object, then release slowly.",
                "Name what boundary was crossed — say it to yourself.",
                "Visualize a safe place where you can speak calmly.",
                "Rephrase your thought starting with ‘I feel... because...’",
                "Do 5 deep squats or stretches — convert energy physically.",
                "Decide to respond only after one deep exhale.",
                "Walk at a slower pace than usual for 2 minutes.",
                "Wash your hands mindfully — symbolic reset."
            ],
            "confused": [
                "Draw your dilemma in two boxes: what’s known vs unknown.",
                "Write a letter to yourself giving advice from tomorrow.",
                "Ask a friend to mirror your thoughts back to you.",
                "List 3 possible next steps — choose the smallest one.",
                "Free-write for 2 minutes — clarity appears mid-sentence.",
                "Take a short walk; new ideas often appear while moving.",
                "Summarize the problem as if explaining to a child.",
                "Pick one option to test for 24 hours and observe.",
                "Sleep before deciding; note what changes in morning.",
                "Replace the word ‘should’ with ‘could’ in your thoughts.",
                "Write your question in one clear sentence.",
                "Look at your situation from a third-person view."
            ],
            "neutral": [
                "Do one mindful act — drink tea slowly or breathe.",
                "Organize a tiny part of your space for freshness.",
                "Write one small thing that went unnoticed but good.",
                "Stretch your hands and neck gently.",
                "Walk 3 minutes outdoors or near a window.",
                "Reflect on one goal quietly for 2 minutes.",
                "Do something creative with no result expectation.",
                "Share a kind word online or in person.",
                "Plan tomorrow with light music.",
                "Read one positive quote before bed.",
                "Water a plant or tidy your desk mindfully.",
                "List 3 ordinary things you’re grateful for."
            ],
            "motivated": [
                "Write the top 3 tasks for today in order.",
                "Start with the easiest 2-minute version of your goal.",
                "Set a 20-minute deep-work timer.",
                "Eliminate one small distraction for an hour.",
                "Visualize the feeling after finishing one key task.",
                "Re-read your reason for starting this journey.",
                "Encourage someone else — momentum grows when shared.",
                "Log your wins for the day, no matter how small.",
                "Take a power stretch after 40 minutes of work.",
                "End the day by planning tomorrow’s first 15 minutes.",
                "Rearrange your workspace slightly for energy refresh.",
                "Say ‘I’m doing enough’ aloud once today."
            ],
            "calm": [
                "Close your eyes and focus on breathing for 60 seconds.",
                "Notice three physical sensations of calm in your body.",
                "Light a candle or look at a soft color mindfully.",
                "Write one line of gratitude for your current peace.",
                "Limit screen time for the next hour.",
                "Walk slowly noticing rhythm of steps.",
                "Avoid multitasking for 30 minutes.",
                "Enjoy silence for one full minute.",
                "Drink something warm without distraction.",
                "Read one paragraph slowly for meaning, not speed.",
                "Stretch slowly with calm music.",
                "Step outside and feel fresh air on your skin."
            ]
        }


    # -------------------------
    # --- lightweight helpers
    # -------------------------
    @staticmethod
    def _sentences(text):
        # naive sentence splitter (works fine for short reflections)
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _keyword_emotion(self, text):
        # crude lexical heuristic to decide mood
        t = text.lower()
        pos = sum(1 for w in re.findall(r'\w+', t) if w in self._positive_words)
        neg = sum(1 for w in re.findall(r'\w+', t) if w in self._negative_words)
        if pos > neg + 1:
            return "optimistic", 0.55
        if neg > pos + 1:
            # if many anxious words present
            if any(x in t for x in ["anx", "worry", "panic", "nerv"]):
                return "anxious", 0.5
            if any(x in t for x in ["angr", "rage", "fury"]):
                return "angry", 0.5
            return "sad", 0.5
        # neutral or confused fallback
        if any(x in t for x in ["confus", "uncertain", "not sure"]):
            return "confused", 0.45
        return "neutral", 0.4

    # Zero-shot classifier wrapper + mapping
    def _zero_shot_mood(self, text):
        """
        If zero-shot classifier exists, use it with clear emotion labels.
        If low confidence, fall back to keyword heuristics.
        Returns: (mood_key, confidence, scores_dict)
        """
        candidate_labels = [
            "joy", "sadness", "anger", "fear", "neutral",
            "confusion", "motivation", "calm", "anxiety", "optimism"
        ]
        label_map = {
            "joy": "optimistic",
            "optimism": "optimistic",
            "sadness": "sad",
            "anger": "angry",
            "fear": "anxious",
            "anxiety": "anxious",
            "confusion": "confused",
            "motivation": "motivated",
            "calm": "calm",
            "neutral": "neutral"
        }

        if self.zero_shot:
            try:
                out = self.zero_shot(text, candidate_labels=candidate_labels, multi_label=False)
                labels = out.get("labels", [])
                scores = out.get("scores", [])
                if labels and scores:
                    top_label = labels[0]
                    top_score = float(scores[0])
                    mapped = label_map.get(top_label, top_label)
                    if top_score >= 0.45:
                        return mapped, top_score, dict(zip(labels, scores))
                    # low confidence -> try keyword fallback
                    fallback_mood, fallback_score = self._keyword_emotion(text)
                    return fallback_mood, top_score, dict(zip(labels, scores))
            except Exception:
                pass

        # no classifier -> keyword
        return self._keyword_emotion(text)

    def _classify_text(self, text):
        """
        Try to use a fine-tuned classifier (text-classification).
        If that returns a label set we map it; otherwise use zero-shot or keyword.
        """
        # 1) fine-tuned text-classifier (single-label)
        if self.classifier:
            try:
                out = self.classifier(text)
                if isinstance(out, list) and out:
                    lab = out[0].get("label", "").lower()
                    score = float(out[0].get("score", 0.0))
                    # try mapping common outputs into our mood keys
                    if "joy" in lab or "happy" in lab or "positive" in lab:
                        return "optimistic", score, {lab: score}
                    if "sad" in lab or "depress" in lab:
                        return "sad", score, {lab: score}
                    if "angry" in lab or "anger" in lab:
                        return "angry", score, {lab: score}
                    if "anx" in lab or "fear" in lab:
                        return "anxious", score, {lab: score}
                    if "neutral" in lab:
                        return "neutral", score, {lab: score}
                    if "confus" in lab:
                        return "confused", score, {lab: score}
                    # fallback to label as-is
                    return lab, score, {lab: score}
            except Exception:
                pass

        # 2) zero-shot or keyword fallback
        return self._zero_shot_mood(text)

    def summarize(self, text):
        # Use pipeline summarizer if available; else naive first-two-sentences
        if self.summarizer:
            try:
                out = self.summarizer(text, max_length=120, min_length=30, do_sample=False)
                if isinstance(out, list) and out:
                    return out[0].get("summary_text", "").strip()
            except Exception:
                pass
        # fallback: first 2 sentences or shorten
        sents = self._sentences(text)
        if not sents:
            return text.strip()[:300]
        return " ".join(sents[:2]) if len(sents) > 1 else sents[0]

    def balance_score(self, text):
        """
        Return a simple 'balance' number 0..100 based on positivity - negativity.
        This is heuristic, intentionally lightweight.
        """
        words = re.findall(r'\w+', text.lower())
        if not words:
            return 50
        pos = sum(1 for w in words if w in self._positive_words)
        neg = sum(1 for w in words if w in self._negative_words)
        raw = pos - neg
        # normalize to 0..100 (centered at 50)
        score = 50 + (raw * 10)
        score = max(0, min(100, int(round(score))))
        return score

    def analyze(self, text: str) -> dict:
        """
        Main analysis function — returns a dictionary with keys:
         - summary (str)
         - mood (str)
         - mood_confidence (float)
         - balance_score (int 0..100)
         - reflection_prompts (list)
         - micro_actions (list)
         - suggestions (list)  (additional helpful suggestions)
        """
        text = text.strip()
        result = {
            "summary": "",
            "mood": "neutral",
            "mood_confidence": 0.0,
            "balance_score": 50,
            "reflection_prompts": [],
            "micro_actions": [],
            "suggestions": [],
        }

        if not text:
            # nothing to analyze
            result["summary"] = ""
            return result

        # summary
        try:
            result["summary"] = self.summarize(text)
        except Exception:
            result["summary"] = "Could not produce a summary."

        # mood classification
        try:
            mood, conf, scores = self._classify_text(text)
            result["mood"] = mood
            result["mood_confidence"] = float(conf)
            result["scores"] = scores
        except Exception:
            # fallback
            mood, conf = self._keyword_emotion(text)
            result["mood"] = mood
            result["mood_confidence"] = float(conf)
            result["scores"] = {}

        # balance score
        try:
            result["balance_score"] = self.balance_score(text)
        except Exception:
            result["balance_score"] = 50

        # reflection prompts and micro-actions
        prompts = self.mood_prompts.get(result["mood"], self.mood_prompts.get("neutral", []))
        actions = self.micro_actions.get(result["mood"], self.micro_actions.get("neutral", []))

        # pick two items each for readability
        result["reflection_prompts"] = random.sample(prompts, min(2, len(prompts)))
        result["micro_actions"] = random.sample(actions, min(2, len(actions)))

        # pragmatic suggestions (safe, brief)
        result["suggestions"] = [
            "If you feel overwhelmed, try one tiny, concrete next step (2–10 minutes).",
            "If the mood persists, consider talking to a trusted person or a professional.",
        ]

        return result
