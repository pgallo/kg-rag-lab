"""RDF namespaces and the small vocabulary of entity types used across the lab."""

from __future__ import annotations

from rdflib import Namespace
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

KG = Namespace("https://example.org/kgrag/ontology#")
ID = Namespace("https://example.org/kgrag/id/")

# Main entity types, in the order used for display and type resolution.
ENTITY_TYPES = (
    KG.Drug,
    KG.Target,
    KG.Disease,
    KG.Pathway,
    KG.ClinicalTrial,
    KG.Publication,
)

# Object properties that carry instance-level facts (the ABox edges).
FACT_PREDICATES = (
    KG.inhibits,
    KG.associatedWith,
    KG.participatesIn,
    KG.studiedIn,
    KG.investigates,
    KG.mentions,
)

__all__ = ["ENTITY_TYPES", "FACT_PREDICATES", "ID", "KG", "OWL", "RDF", "RDFS", "SKOS", "XSD"]
