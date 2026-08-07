"""
resources_data.py — structured version of Lost_Voices_Resource_Bank.docx,
grouped by capstone deliverable, for rendering on the Resources page.
"""

DELIVERABLES = [
    {
        "title": "Deliverable 1 — Language Tech Landscape",
        "groups": [
            {
                "name": "Mozilla Common Voice",
                "links": [
                    {"label": "Foundational paper (Ardila et al., LREC 2020) — ACL Anthology", "url": "https://aclanthology.org/2020.lrec-1.520.pdf"},
                    {"label": "Same paper — arXiv version", "url": "https://arxiv.org/abs/1912.06670"},
                    {"label": "Official site", "url": "https://commonvoice.mozilla.org"},
                    {"label": "Dataset/versioning repo", "url": "https://github.com/common-voice/cv-dataset"},
                    {"label": "Community strategy wiki", "url": "https://m.wiki.mozilla.org/CommonVoice"},
                    {"label": "GIZ/Mozilla strategy reflection report (East Africa deployments)", "url": "https://www.bmz-digital.global/wp-content/uploads/2023/03/Creating-Community-Driven-Datasets-Report-032023-GIZ-Mozilla.pdf"},
                    {"label": "Data quality issues in multilingual speech datasets (sociolinguistic checklist)", "url": "https://arxiv.org/html/2506.17525v1"},
                    {"label": "Pashto Common Voice case study", "url": "https://arxiv.org/abs/2603.27021"},
                ],
                "use_for": "Crowdsourcing model mechanics, CC0 licensing contrast, limitations argument (population/infrastructure dependency, Wikipedia-sourcing barrier).",
            },
            {
                "name": "AI4Bharat",
                "links": [
                    {"label": "Main site", "url": "https://ai4bharat.iitm.ac.in"},
                    {"label": "TTS area page", "url": "https://ai4bharat.iitm.ac.in/areas/tts"},
                    {"label": "Rasa dataset (HuggingFace)", "url": "https://huggingface.co/datasets/ai4bharat/Rasa"},
                    {"label": "IndicVoices paper", "url": "https://arxiv.org/abs/2403.01926"},
                    {"label": "Indic-TTS GitHub repo", "url": "https://github.com/AI4Bharat/Indic-TTS"},
                    {"label": "Bhashini (Govt of India National Language Translation Mission)", "url": "https://bhashini.gov.in"},
                    {"label": "IndicOOV paper — OOV performance via low-effort data strategies", "url": "https://arxiv.org/abs/2407.13435"},
                    {"label": "ASR pseudo-labeling paper", "url": "https://arxiv.org/abs/2408.14026"},
                ],
                "use_for": "Institutional/government-funded pipeline model, standardisation-over-volume finding, funding precision (funded BY Bhashini, not run under it), scheduled-language scope limitation.",
            },
        ],
    },
    {
        "title": "Deliverable 2 — Archive-to-Platform Strategy",
        "groups": [
            {
                "name": "ELAR (Endangered Languages Archive)",
                "links": [
                    {"label": "Main catalogue", "url": "https://www.elararchive.org/"},
                    {"label": "Background/history", "url": "https://en.wikipedia.org/wiki/Endangered_Languages_Archive"},
                    {"label": "Access tiers explainer", "url": "https://www.elar-archive.org/all-courses/"},
                ],
                "use_for": "The O/U/S tiered access mechanism — borrowed for Lost Voices' Open/Registered/Reviewed access tiers.",
            },
            {
                "name": "PARADISEC",
                "links": [
                    {"label": "Main site", "url": "https://www.paradisec.org.au/"},
                    {"label": "Catalogue/viewer", "url": "https://catalog.paradisec.org.au/"},
                    {"label": "Background", "url": "https://en.wikipedia.org/wiki/PARADISEC"},
                    {"label": "New catalog viewer blog post (RO-Crate modernisation)", "url": "https://www.paradisec.org.au/blog/2026/02/the-new-paradisec-catalog-viewer/"},
                ],
                "use_for": "Community-return motivation, live example of archive-to-platform modernisation in progress.",
            },
            {
                "name": "Te Hiku Media / Papa Reo",
                "links": [
                    {"label": "Papa Reo platform", "url": "https://papareo.nz/"},
                    {"label": "Kaitiakitanga License explainer", "url": "https://tehiku.nz/te-hiku-tech/te-hiku-dev-korero/25141/data-sovereignty-and-the-kaitiakitanga-license"},
                    {"label": "MIT Technology Review feature", "url": "https://www.technologyreview.com/2022/04/22/1050394/artificial-intelligence-for-the-people/"},
                    {"label": "NVIDIA blog (NeMo, technical approach)", "url": "https://blogs.nvidia.com/blog/te-hiku-media-maori-speech-ai/"},
                ],
                "use_for": "Living-platform model (vs. archive), Kaitiakitanga governance philosophy, dedicated-model-per-language + reusable-toolkit scaling approach — directly informs Deliverables 3 and 4.",
            },
        ],
    },
    {
        "title": "Deliverable 3 — AI Technology Stack",
        "groups": [
            {
                "name": "Core architecture papers",
                "links": [
                    {"label": "MMS — Scaling Speech Technology to 1,000+ Languages (Pratap et al.)", "url": "https://arxiv.org/abs/2305.13516"},
                    {"label": "NLLB-200 — No Language Left Behind (Meta)", "url": "https://arxiv.org/abs/2207.04672"},
                    {"label": "MAD-X — adapter-based multilingual framework (Pfeiffer et al.)", "url": "https://arxiv.org/abs/2005.00052"},
                    {"label": "AdapterHub — practical adapter toolkit", "url": "https://adapterhub.ml"},
                    {"label": "Small Models, Big Impact — survey of adapter-based methods", "url": "https://arxiv.org/abs/2502.10140"},
                    {"label": "IndicTrans2 GitHub (script unification, shared backbone)", "url": "https://github.com/AI4Bharat/IndicTrans2"},
                ],
                "use_for": "The three-way scaling architecture comparison (shared-backbone vs. adapters vs. dedicated-model-per-language) that anchors the Lost Voices architecture decision.",
            },
        ],
    },
    {
        "title": "Deliverable 4 — Community Consent & Ownership",
        "groups": [
            {
                "name": "CARE Principles for Indigenous Data Governance",
                "links": [
                    {"label": "Main page (Global Indigenous Data Alliance)", "url": "https://www.gida-global.org/care-principles-copy"},
                    {"label": "Original publication (Carroll et al., 2020, Data Science Journal)", "url": "https://datascience.codata.org/articles/10.5334/dsj-2020-043"},
                    {"label": "Overview", "url": "https://en.wikipedia.org/wiki/CARE_Principles_for_Indigenous_Data_Governance"},
                    {"label": "\"It's How You Do Things That Matters\" — core reading", "url": "https://arxiv.org/abs/2402.02639"},
                ],
                "use_for": "Synthesises CARE, Te Mana Raraunga, and Maiam nayri Wingara principles specifically for language technology projects — the single most directly relevant paper for this deliverable.",
            },
            {
                "name": "OCAP (Ownership, Control, Access, Possession)",
                "links": [
                    {"label": "FNIGC OCAP training/overview", "url": "https://fnigc.ca/ocap-training/"},
                    {"label": "Wikipedia summary", "url": "https://en.wikipedia.org/wiki/First_Nations_principles_of_OCAP"},
                ],
                "use_for": "The ownership/possession distinction — Lost Voices as steward/possessor, Sunuwar Welfare Society as owner/controller.",
            },
            {
                "name": "Local Contexts / Traditional Knowledge (TK) Labels",
                "links": [
                    {"label": "Main site", "url": "https://localcontexts.org/"},
                    {"label": "TK Labels explainer", "url": "https://localcontexts.org/labels/traditional-knowledge-labels/"},
                    {"label": "Mukurtu FAQ (practical implementation detail)", "url": "https://mukurtu.org/support/traditional-knowledge-labels-faq/"},
                ],
                "use_for": "The practical, per-item metadata labelling mechanism underneath the governance framework.",
            },
        ],
    },
    {
        "title": "Deliverable 5 — Revenue Model",
        "groups": [
            {
                "name": "Lelapa AI / Vulavula (primary comparable)",
                "links": [
                    {"label": "Pricing page", "url": "https://lelapa.ai/pricing"},
                    {"label": "Pricing detail page", "url": "https://lelapa.ai/about/pricing/"},
                    {"label": "Vulavula API docs", "url": "https://docs.lelapa.ai/"},
                    {"label": "Main site", "url": "https://lelapa.ai/"},
                ],
                "use_for": "Confirmed live tiered subscription + usage-based API pricing model — direct evidence for the Stream 1 (API/infrastructure licensing) Year-1 recommendation.",
            },
            {
                "name": "Duolingo Hawaiian/Navajo — cautionary precedent",
                "links": [
                    {"label": "CNN feature via All Things Linguistic (community reception issues)", "url": "https://allthingslinguistic.com/post/188223483472/duolingo-and-smaller-languages-useful-but-also"},
                ],
                "use_for": "Documented failure mode (content built without community co-design) — justifies treating the diaspora subscription stream as higher-risk and Year-2, not Year-1.",
            },
        ],
    },
    {
        "title": "Cross-Cutting / Background Material",
        "groups": [
            {
                "name": "Referenced during technology-stack discussions",
                "links": [
                    {"label": "A2TTS — speaker-conditioned diffusion TTS for low-resource Indian languages (IIT Bombay)", "url": "https://arxiv.org/abs/2507.15272"},
                ],
                "use_for": "Not directly adopted (different architecture/compute/speaker-diversity assumptions), but useful for the CER-vs-WER evaluation argument used in the TTS evaluation discussion.",
            },
        ],
    },
]

OUR_PAPER = {
    "status": "coming_soon",
    "title": "Lost Voices: A Monolingual NLP and Text-to-Speech Pipeline for Sunuwar",
    "note": "Our own paper will be linked here once the final report (Phase 7-8 evaluation) is complete.",
}
