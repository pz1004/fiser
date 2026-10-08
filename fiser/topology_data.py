"""Benchmark topologies with link lengths in km (NSFNET, COST239, USNET).

Plain data module (standard library only). Each topology is a dict with keys:
    name   -- short name
    nodes  -- node names in index order
    edges  -- list of (u, v, km) tuples, 0-indexed, undirected
    coords -- dict index -> (lat, lon) in decimal degrees, or None
    source -- citation + URL for the link lengths
    note   -- provenance caveats; read before citing

Compiled 2026-10-07. Sources were read directly and checked with networkx
(see each topology's source and note fields). Coordinates are
approximate city centres supplied by us. They are NOT from the length sources
and are meant only for satellite-visibility geometry.
"""

NSFNET = {
    "name": "NSFNET",
    "nodes": [
        "Seattle, WA",           # 0  (WA)
        "Palo Alto, CA",         # 1  (CA1)
        "San Diego, CA",         # 2  (CA2)
        "Salt Lake City, UT",    # 3  (UT)
        "Boulder, CO",           # 4  (CO)
        "Houston, TX",           # 5  (TX)
        "Lincoln, NE",           # 6  (NE)
        "Champaign, IL",         # 7  (IL)
        "Pittsburgh, PA",        # 8  (PA)
        "Atlanta, GA",           # 9  (GA)
        "Ann Arbor, MI",         # 10 (MI)
        "Ithaca, NY",            # 11 (NY)
        "Princeton, NJ",         # 12 (NJ)
        "College Park, MD",      # 13 (MD; some files label it DC)
    ],
    "edges": [
        (0, 1, 2100), (0, 2, 3000), (0, 7, 4800), (1, 2, 1200), (1, 3, 1500),
        (2, 5, 3600), (3, 4, 1200), (3, 10, 3900), (4, 5, 2400), (4, 6, 1200),
        (5, 9, 2100), (5, 13, 3600), (6, 7, 1500), (7, 8, 1500), (8, 9, 1500),
        (8, 11, 600), (8, 12, 600), (10, 11, 1200), (10, 12, 1500),
        (11, 13, 600), (12, 13, 300),
    ],
    "coords": {
        0: (47.606, -122.332), 1: (37.442, -122.143), 2: (32.716, -117.161),
        3: (40.761, -111.891), 4: (40.015, -105.271), 5: (29.760, -95.370),
        6: (40.814, -96.703), 7: (40.116, -88.243), 8: (40.441, -79.996),
        9: (33.749, -84.388), 10: (42.281, -83.743), 11: (42.444, -76.502),
        12: (40.357, -74.667), 13: (38.981, -76.937),
    },
    "source": (
        "Link lengths: NSF.m in the DeepRMSA code release of X. Chen et al., "
        "'DeepRMSA: A Deep Reinforcement Learning Framework for Routing, Modulation "
        "and Spectrum Assignment in Elastic Optical Networks', J. Lightw. Technol. "
        "37(16):4155-4163, 2019, doi:10.1109/JLT.2019.2923615. File: "
        "https://github.com/xiaoliangchenUCD/DeepRMSA/blob/master/NSF.m "
        "(14x14 km matrix, read before its 'topology = topology/2' line)."
    ),
    "note": (
        "All 21 (u, v, km) values match the NSF.m matrix exactly. NSF.m has one "
        "more link, NE-GA = (6, 9, 2700), which gives the 22-link variant. "
        "DeepRMSA itself halves the matrix (its comment dated 2013-7-18 says to "
        "halve the distances), and so do optical-rl-gym's nsfnet_chen.txt and "
        "XLRON's nsfnet_deeprmsa: they use 22 links at half length. Different "
        "NSFNET length sets exist elsewhere (e.g. Ghose et al. 2005, used in "
        "arXiv:2307.04477), so the paper must say which set it uses. NSF.m has "
        "no node names. City names are the conventional NSFNET T1 sites, "
        "inferred by us. Every link is 1.2-2.1x the great-circle distance "
        "between these cities, which is plausible for fibre routes. Total "
        "length 39,900 km; mean link 1,900 km."
    ),
}

COST239 = {
    "name": "COST239",
    "nodes": [
        "London",      # 0
        "Paris",       # 1
        "Brussels",    # 2
        "Amsterdam",   # 3
        "Luxembourg",  # 4
        "Zurich",      # 5
        "Milan",       # 6
        "Copenhagen",  # 7
        "Berlin",      # 8
        "Prague",      # 9
        "Vienna",      # 10
    ],
    "edges": [
        (0, 1, 450), (0, 2, 390), (0, 3, 550), (0, 7, 1310),
        (1, 2, 300), (1, 4, 400), (1, 5, 600), (1, 6, 820), (1, 8, 1090),
        (2, 3, 210), (2, 4, 220), (2, 6, 930),
        (3, 4, 390), (3, 7, 760), (3, 8, 660),
        (4, 5, 350), (4, 9, 730),
        (5, 6, 320), (5, 9, 565), (5, 10, 730),
        (6, 10, 820),
        (7, 8, 390), (7, 9, 740),
        (8, 9, 340), (8, 10, 660),
        (9, 10, 320),
    ],
    "coords": {
        0: (51.507, -0.128), 1: (48.857, 2.352), 2: (50.850, 4.352),
        3: (52.370, 4.895), 4: (49.612, 6.130), 5: (47.377, 8.541),
        6: (45.464, 9.190), 7: (55.676, 12.568), 8: (52.520, 13.405),
        9: (50.076, 14.438), 10: (48.208, 16.373),
    },
    "source": (
        "Link lengths: Cost_239.m in the DeepRMSA code release (X. Chen et al., "
        "J. Lightw. Technol. 37(16):4155-4163, 2019, doi:10.1109/JLT.2019.2923615), "
        "https://github.com/xiaoliangchenUCD/DeepRMSA/blob/master/Cost_239.m "
        "(values are written there as 'km*2'; we list the km values). "
        "Cross-check: S. N. Avci and E. Ayanoglu, 'Coded Path Protection Part 2', "
        "arXiv:1208.5528, Fig. 2 'European COST 239 network' "
        "(https://arxiv.org/abs/1208.5528), which cites Batchelor et al., "
        "'Ultra high capacity optical transmission networks: final report of "
        "action COST 239', 1999."
    ),
    "note": (
        "Node order follows the numbering in Cost_239.m. Neither source names the "
        "nodes. We assigned the 11 standard COST239 cities by geography: every "
        "link is then 1.07-1.54x the great-circle distance between its cities. "
        "The Avci-Ayanoglu figure agrees on 24 of its 25 labelled links. It gives "
        "Brussels-Milan as 920 km (Cost_239.m: 930) and leaves Luxembourg-Prague "
        "unlabelled (Cost_239.m: 730). XLRON's 'cost239_ptrnet_published' "
        "variant differs on four links: Brussels-Paris 294, Luxembourg-Paris 350, "
        "Brussels-Milan 900, Prague-Zurich 560. Its 'ptrnet_real' variant swaps "
        "Brussels-Luxembourg and Brussels-Paris to 294 and 220, and "
        "Luxembourg-Prague and Luxembourg-Paris to 350 and 730. Those swapped "
        "values are shorter than the straight-line distances, so they are "
        "physically impossible. DeepRMSA's own runs used doubled lengths."
    ),
}

USNET = {
    "name": "USNET",
    "nodes": ["N%d" % i for i in range(1, 25)],  # index k is node k+1 of the source figure
    "edges": [
        (0, 1, 800), (0, 5, 1000), (1, 2, 1100), (1, 5, 950),
        (2, 3, 250), (2, 4, 1000), (2, 6, 1000), (3, 4, 800), (3, 6, 850),
        (4, 7, 1200), (5, 6, 1000), (5, 8, 1200), (5, 10, 1900),
        (6, 7, 1150), (6, 8, 1000), (7, 9, 900), (8, 9, 1000),
        (8, 10, 1400), (8, 11, 1000), (9, 12, 950), (9, 13, 850),
        (10, 11, 900), (10, 14, 1300), (10, 18, 2600),
        (11, 12, 900), (11, 15, 1000), (12, 13, 650), (12, 16, 1100),
        (13, 17, 1200), (14, 15, 600), (14, 19, 1300),
        (15, 16, 1000), (15, 20, 1000), (15, 21, 800),
        (16, 17, 800), (16, 21, 850), (16, 22, 1000), (17, 23, 900),
        (18, 19, 1200), (19, 20, 700), (20, 21, 300), (21, 22, 600),
        (22, 23, 900),
    ],
    "coords": None,
    "source": (
        "Link lengths read from Fig. 2.2 'USNET topology' (thesis p. 31) of "
        "'Research on Survivability Strategies of Virtual Network', CQUPT master's "
        "thesis, arXiv:2008.07255 (https://arxiv.org/abs/2008.07255). The text "
        "says 24 nodes and 43 bidirectional links, labelled with distances in km."
    ),
    "note": (
        "No peer-reviewed table of USNET lengths was found, so this source is a "
        "master's thesis; cite it with that caveat. XLRON's 'usnet_ptrnet' file "
        "(https://github.com/micdoh/XLRON, xlron/data/topologies/"
        "usnet_ptrnet_undirected.json) is attributed to PtrNet-RSA (Cheng et al., "
        "JLT 2024, doi:10.1109/JLT.2024.3405587, paywalled, not checked). It has "
        "the same 43 links and numbering but differs on 5 lengths (0-indexed): "
        "(10,18) 2800, (11,12) 1000, (11,15) 1100, (14,15) 800, (18,19) 700. "
        "Neither source gives node cities. XLRON's USNET coordinates are its own "
        "placements. With the lengths above, 22 of the 43 links come out shorter "
        "than the straight-line distance between those coordinates, or more than "
        "twice it, so we do not use them (coords = None)."
    ),
}

# Machine-readable record of the known alternative values discussed in the notes
# (0-indexed (u, v) -> km). Use for sensitivity checks; the dicts above are the defaults.
ALTERNATIVE_LENGTHS = {
    "NSFNET_extra_link_22link_variant": {(6, 9): 2700},
    "COST239_avci_ayanoglu_fig2": {(2, 6): 920},
    "USNET_xlron_ptrnet": {(10, 18): 2800, (11, 12): 1000, (11, 15): 1100,
                           (14, 15): 800, (18, 19): 700},
}

TOPOLOGIES = {"NSFNET": NSFNET, "COST239": COST239, "USNET": USNET}
