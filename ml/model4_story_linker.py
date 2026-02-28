"""
MigrateIQ — Model 4: Story Event Linker (RAG + LLM)
====================================================
Connects real-world environmental events (news, climate data,
policy changes) to detected migration anomalies and generates
human-readable cause-effect story cards for the public feed.

Pipeline:
  1. Anomaly detected by Model 3  →
  2. Retrieve relevant events from vector store (RAG)  →
  3. Claude API generates a public-friendly story card  →
  4. Story card returned to frontend

Architecture:
  - Vector store: FAISS for semantic retrieval over news/event corpus
  - Embeddings: sentence-transformers (all-MiniLM-L6-v2)
  - LLM: Claude API (claude-sonnet) for story generation

Install: pip install anthropic faiss-cpu sentence-transformers numpy
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

# ─── optional imports with graceful fallback ────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("faiss not installed — using simple cosine search fallback")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("sentence-transformers not installed — using dummy embeddings")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("anthropic not installed — story generation will use template fallback")




@dataclass
class WorldEvent:
    """A real-world event that might affect animal migration."""
    event_id:    str
    title:       str
    date:        str          # ISO format YYYY-MM-DD
    location:    str          # country or region
    event_type:  str          # 'wildfire' | 'drought' | 'flood' | 'heatwave' |
                              # 'deforestation' | 'pollution' | 'policy' | 'other'
    description: str
    affected_species: list[str]    # known affected species
    source:      str          # news outlet / scientific paper
    severity:    str          # 'low' | 'medium' | 'high' | 'critical'


@dataclass
class StoryCard:
    """Public-facing story card for the MigrateIQ feed."""
    card_id:        str
    story_type:     str       # 'event_link' | 'prediction' | 'fact' | 'alert'
    title:          str
    body:           str
    cause:          str       # short cause phrase
    effect:         str       # short effect phrase
    species:        list[str]
    date:           str
    impact_level:   str       # 'low' | 'medium' | 'high'
    confidence:     float     # 0–1
    source_events:  list[str] # event_ids used
    icon:           str       # emoji for UI


# ─────────────────────────────────────────────
# 2. EVENT CORPUS (sample — replace with live news API)
# ─────────────────────────────────────────────

SAMPLE_EVENTS: list[WorldEvent] = [
    WorldEvent('E001', '2023 Amazon Mega-Drought', '2023-08-15', 'Brazil',
               'drought', 'The Amazon basin experienced its worst dry season in 45 years, '
               'with river levels at historic lows and widespread wetland desiccation.',
               ['harpy_eagle', 'giant_river_otter', 'migratory_waterfowl'], 'Reuters', 'critical'),

    WorldEvent('E002', 'California Wildfire Season 2023', '2023-09-01', 'California, USA',
               'wildfire', 'Over 1.2 million acres burned across central California, '
               'destroying critical overwintering monarch butterfly habitat and '
               'milkweed corridors used as stopover points.',
               ['monarch_butterfly'], 'AP News', 'high'),

    WorldEvent('E003', 'North Atlantic SST Anomaly 2023', '2023-07-01', 'North Atlantic Ocean',
               'heatwave', 'Sea surface temperatures in the North Atlantic hit record highs, '
               'averaging 1.7°C above the 1982–2011 baseline. Fish distributions shifted '
               'poleward, affecting seabird and whale feeding grounds.',
               ['arctic_tern', 'humpback_whale', 'puffin'], 'NOAA', 'critical'),

    WorldEvent('E004', 'Serengeti Road Development Proposal', '2025-01-10', 'Tanzania',
               'deforestation', 'Plans for a commercial highway through the Serengeti '
               'ecosystem risk bisecting the Great Wildebeest Migration corridor, '
               'potentially fragmenting 300,000 animals from seasonal grazing grounds.',
               ['wildebeest', 'zebra'], 'BBC Wildlife', 'high'),

    WorldEvent('E005', 'Greenland Ice Sheet Acceleration 2022-2024', '2024-03-20',
               'Greenland', 'heatwave',
               'Mass loss from the Greenland Ice Sheet accelerated by 30% compared '
               'to the previous decade, altering Arctic atmospheric circulation '
               'patterns and jet stream behaviour.',
               ['arctic_tern', 'snow_goose', 'bar_tailed_godwit'], 'Nature Climate Change', 'critical'),

    WorldEvent('E006', 'Midwest Milkweed Habitat Loss', '2023-05-01', 'Midwest USA',
               'deforestation', 'Agricultural expansion reduced milkweed populations '
               'by an estimated 58% across the US Midwest over the past two decades, '
               'critically impacting monarch butterfly breeding grounds.',
               ['monarch_butterfly'], 'USFWS Report', 'high'),

    WorldEvent('E007', 'EU Bird Protection Directive Strengthened', '2024-11-01', 'Europe',
               'policy', 'Updated EU directive expands protected flyway corridors '
               'for migratory birds by 18%, offering increased legal protection '
               'to key staging areas along the East Atlantic Flyway.',
               ['arctic_tern', 'white_stork', 'osprey'], 'EU Commission', 'medium'),

    WorldEvent('E008', 'Indian Ocean Marine Heatwave 2024', '2024-02-15', 'Indian Ocean',
               'heatwave', 'A sustained marine heatwave in the southern Indian Ocean '
               'bleached 40% of surveyed coral reefs and displaced krill populations '
               'poleward, affecting humpback whale feeding aggregations.',
               ['humpback_whale', 'southern_right_whale'], 'Marine Conservation Society', 'high'),
]



class EventVectorStore:
    """
    Embeds world events into a vector store for semantic retrieval.
    Uses FAISS for fast approximate nearest-neighbour search.
    Falls back to cosine similarity if FAISS unavailable.
    """

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.events: list[WorldEvent] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.dim   = 384   # MiniLM output dimension

        if ST_AVAILABLE:
            print(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
        else:
            self.embedder = None
            print("Using dummy embeddings (sentence-transformers not installed)")

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.embedder:
            return self.embedder.encode(texts, normalize_embeddings=True,
                                        show_progress_bar=False)
        # Dummy: random unit vectors
        vecs = np.random.randn(len(texts), self.dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def build(self, events: list[WorldEvent]):
        self.events = events
        texts = [f"{e.title}. {e.description}" for e in events]
        self.embeddings = self._embed(texts)

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dim)   # Inner product (cosine on normalised)
            self.index.add(self.embeddings)
            print(f"FAISS index built with {len(events)} events.")
        else:
            print(f"Simple cosine store built with {len(events)} events.")

    def search(self, query: str, k: int = 3,
               species_filter: Optional[str] = None) -> list[tuple[WorldEvent, float]]:
        """
        Semantic search for events relevant to a query.
        Optional species filter narrows to events known to affect that species.
        """
        query_vec = self._embed([query])

        if FAISS_AVAILABLE and self.index:
            scores, idxs = self.index.search(query_vec, min(k * 3, len(self.events)))
            candidates = [(self.events[i], float(scores[0][j]))
                          for j, i in enumerate(idxs[0]) if i >= 0]
        else:
            # Cosine similarity fallback
            sims = (self.embeddings @ query_vec.T).ravel()
            top_k = np.argsort(sims)[::-1][:k*3]
            candidates = [(self.events[i], float(sims[i])) for i in top_k]

        # Optional species filter
        if species_filter:
            candidates = [
                (e, s) for e, s in candidates
                if any(species_filter.lower() in sp.lower()
                       for sp in e.affected_species)
            ] + [
                (e, s * 0.8) for e, s in candidates
                if not any(species_filter.lower() in sp.lower()
                           for sp in e.affected_species)
            ]

        return candidates[:k]



class StoryGenerator:
    """
    Uses Claude API to generate public-friendly story cards
    linking world events to migration anomalies.
    """

    SYSTEM_PROMPT = """You are MigrateIQ's science communicator. 
Your job is to write clear, engaging story cards for the general public 
explaining how real-world events have caused or may cause changes 
in animal migration patterns.

Rules:
- Write for a curious non-scientist (age 14+)
- Be factual and grounded — do not speculate beyond the evidence
- Lead with impact, not methodology
- Keep body text under 60 words
- Keep cause and effect phrases under 10 words each
- Tone: serious but accessible, not alarmist

Always respond with valid JSON only. No preamble, no markdown fences."""

    USER_TEMPLATE = """
An anomaly was detected in the {species} migration for season {year}.
Anomaly details: {anomaly_description}

Related world events:
{events_text}

Generate a story card as JSON with these exact keys:
{{
  "title": "...",         // punchy headline, max 12 words
  "body": "...",          // 40-60 word explanation for general public
  "cause": "...",         // short cause phrase, max 10 words
  "effect": "...",        // short effect phrase, max 10 words
  "impact_level": "...",  // "low" | "medium" | "high"
  "confidence": 0.0,      // float 0-1, how confident is this link
  "icon": "..."           // single relevant emoji
}}
"""

    def __init__(self, api_key: Optional[str] = None):
        if ANTHROPIC_AVAILABLE:
            key = api_key or os.environ.get('ANTHROPIC_API_KEY')
            if key:
                self.client = anthropic.Anthropic(api_key=key)
            else:
                print("Warning: ANTHROPIC_API_KEY not set — using template fallback")
                self.client = None
        else:
            self.client = None

    def generate(self, species: str, year: int,
                 anomaly_description: str,
                 retrieved_events: list[WorldEvent]) -> dict:
        """
        Generates a story card dict using Claude.
        Falls back to template if API unavailable.
        """
        if self.client is None:
            return self._template_fallback(species, year, anomaly_description, retrieved_events)

        events_text = '\n'.join([
            f"- [{e.date}] {e.title}: {e.description[:200]}"
            for e in retrieved_events
        ])

        prompt = self.USER_TEMPLATE.format(
            species=species, year=year,
            anomaly_description=anomaly_description,
            events_text=events_text
        )

        try:
            response = self.client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=400,
                system=self.SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw = response.content[0].text.strip()
            # Strip any accidental markdown fences
            raw = raw.replace('```json', '').replace('```', '').strip()
            return json.loads(raw)
        except Exception as e:
            print(f"Claude API error: {e} — falling back to template")
            return self._template_fallback(species, year, anomaly_description, retrieved_events)

    def _template_fallback(self, species, year, anomaly_description, events):
        """Simple template when API unavailable."""
        top_event = events[0] if events else None
        return {
            'title':        f"{species.replace('_',' ').title()} — {year} Season Anomaly Detected",
            'body':         f"The {year} migration season showed unusual patterns. "
                            f"{anomaly_description} "
                            + (f"This may be linked to: {top_event.title}." if top_event else ''),
            'cause':        top_event.title[:60] if top_event else 'Environmental change detected',
            'effect':       f"Unusual {species} migration pattern in {year}",
            'impact_level': top_event.severity if top_event else 'medium',
            'confidence':   0.65,
            'icon':         '🔍',
        }



class MigrationStoryPipeline:
    """
    End-to-end pipeline: anomaly signal → story card.

    Usage:
        pipeline = MigrationStoryPipeline()
        pipeline.build_index(events)
        card = pipeline.generate_story(anomaly_result)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.store     = EventVectorStore()
        self.generator = StoryGenerator(api_key=api_key)

    def build_index(self, events: list[WorldEvent] = None):
        """Load events into the vector store."""
        events = events or SAMPLE_EVENTS
        self.store.build(events)

    def generate_story(self, anomaly) -> StoryCard:
        """
        Takes an AnomalyResult (from Model 3) and generates a StoryCard.
        Works with both AnomalyResult objects and plain dicts.
        """
        # Support plain dict input too
        if isinstance(anomaly, dict):
            species      = anomaly.get('species', 'unknown')
            year         = anomaly.get('season_year', 2024)
            description  = anomaly.get('story_trigger', 'Anomaly detected.')
            anom_types   = anomaly.get('anomaly_type', [])
        else:
            species      = anomaly.species
            year         = anomaly.season_year
            description  = anomaly.story_trigger or 'Unusual migration pattern detected.'
            anom_types   = anomaly.anomaly_type

        # Build retrieval query
        query = f"{species} migration change {' '.join(anom_types)} {year}"

        # Retrieve top 3 relevant events
        results = self.store.search(query, k=3, species_filter=species)
        top_events  = [e for e, _ in results]
        top_scores  = [s for _, s in results]

        # Generate story with LLM
        card_data = self.generator.generate(species, year, description, top_events)

        import uuid, datetime
        card = StoryCard(
            card_id       = str(uuid.uuid4())[:8],
            story_type    = 'event_link',
            title         = card_data.get('title', 'Migration Alert'),
            body          = card_data.get('body', description),
            cause         = card_data.get('cause', ''),
            effect        = card_data.get('effect', ''),
            species       = [species],
            date          = datetime.date.today().isoformat(),
            impact_level  = card_data.get('impact_level', 'medium'),
            confidence    = card_data.get('confidence', 0.7),
            source_events = [e.event_id for e in top_events],
            icon          = card_data.get('icon', '📍'),
        )
        return card

    def batch_generate(self, anomalies: list) -> list[StoryCard]:
        """Process a batch of anomalies into story cards."""
        cards = []
        for anomaly in anomalies:
            if (isinstance(anomaly, dict) and anomaly.get('is_anomaly', False)) or \
               (hasattr(anomaly, 'is_anomaly') and anomaly.is_anomaly):
                card = self.generate_story(anomaly)
                cards.append(card)
        print(f"Generated {len(cards)} story cards from {len(anomalies)} anomalies.")
        return cards



if __name__ == '__main__':
    print("=" * 60)
    print("MigrateIQ — Story Event Linker (RAG + LLM) — Quickstart")
    print("=" * 60)

    # 1. Build pipeline
    pipeline = MigrationStoryPipeline(
        api_key=os.environ.get('ANTHROPIC_API_KEY')
    )
    pipeline.build_index(SAMPLE_EVENTS)

    # 2. Simulate anomaly results from Model 3
    test_anomalies = [
        {
            'species':      'arctic_tern',
            'season_year':  2023,
            'is_anomaly':   True,
            'anomaly_type': ['route_deviation', 'timing_shift'],
            'story_trigger': 'Arctic Tern 2023 season: route deviated 120km east, '
                             'migration started 18 days later than historical average. '
                             'Temperature anomaly correlated with route change.',
            'anomaly_score': 0.87,
            'severity':     'severe',
        },
        {
            'species':      'monarch_butterfly',
            'season_year':  2023,
            'is_anomaly':   True,
            'anomaly_type': ['timing_shift', 'habitat_degradation'],
            'story_trigger': 'Monarch Butterfly 2023: arrival 5 weeks late, '
                             'NDVI drop along central route suggests milkweed shortage.',
            'anomaly_score': 0.91,
            'severity':     'severe',
        },
        {
            'species':      'wildebeest',
            'season_year':  2024,
            'is_anomaly':   True,
            'anomaly_type': ['route_deviation'],
            'story_trigger': 'Wildebeest 2024: corridor narrowing detected in northern sector.',
            'anomaly_score': 0.62,
            'severity':     'moderate',
        },
        {
            'species':      'humpback_whale',
            'season_year':  2020,
            'is_anomaly':   False,   # ← will be skipped
            'anomaly_type': [],
            'story_trigger': None,
        }
    ]

    # 3. Generate story cards
    cards = pipeline.batch_generate(test_anomalies)

    print("\n── Generated Story Cards ──\n")
    for card in cards:
        print(f"{'═'*56}")
        print(f"  {card.icon}  {card.title}")
        print(f"  Species : {', '.join(card.species)}")
        print(f"  Body    : {card.body}")
        print(f"  Cause   : {card.cause}")
        print(f"  Effect  : {card.effect}")
        print(f"  Impact  : {card.impact_level.upper()} | Confidence: {card.confidence:.0%}")
        print(f"  Sources : {card.source_events}")
        print()

    # 4. Also demo direct semantic search
    print("\n── Semantic Event Search ──")
    query = "ocean warming seabird food disruption"
    results = pipeline.store.search(query, k=3)
    print(f"Query: '{query}'")
    for event, score in results:
        print(f"  {score:.3f} — {event.title} ({event.date})")
