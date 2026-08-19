Treat the following as the long-term product and architecture direction for this project.

## North Star

I am not building a one-off RAG application.

I am building a **reusable, production-grade private knowledge foundation** that can later be verticalized into specialized B2B products and client projects.

The commercial objective is to build this foundation as a reusable engineering asset: invest heavily in the core once, then reuse and adapt it across multiple client projects and vertical products, including international/freelance work. Architectural decisions should therefore favor reusable capabilities, shorter future delivery cycles, strong delivery margins, and a codebase that can be adapted without unnecessarily forking the core.

The long-term model is:

```text
Reusable Foundation
├── High-quality document ingestion
│   ├── PDF
│   ├── DOCX
│   ├── Persian
│   ├── English
│   └── Mixed-language
├── Retrieval foundation
│   ├── modular retrieval boundary
│   ├── semantic retrieval
│   └── hybrid/lexical retrieval only when justified by evidence
├── Grounded generation
│   ├── citations
│   ├── refusal / insufficient-context handling
│   └── provider-independent LLM integration
├── Privacy-conscious / local-first capability
├── Windows-first deployment
├── Linux/server deployment profile
├── reliable indexing and recovery
└── simple customer-facing UX
          ↓
      Vertical layer
          ↓
┌──────────────────────────────────────┐
│ Tender / RFP Intelligence             │
│ Contract / Legal Intelligence         │
│ Technical Documentation Assistant     │
│ Research Assistant                    │
└──────────────────────────────────────┘
```

These are examples of potentially valuable future verticals, not a fixed roadmap. Choose future verticals based on real customer demand and commercial evidence.

## Strategic principle

The foundation should become a reusable engineering asset.

A future project should ideally reuse most of:

* ingestion
* parsing
* normalization
* chunking
* metadata
* embeddings interface
* vector storage
* retrieval boundary
* grounding
* citations
* provider abstraction
* evaluation infrastructure
* deployment infrastructure
* error handling
* observability and configuration

while the client-specific layer primarily adds:

* domain models
* business rules
* workflows
* specialized prompts
* domain-specific retrieval/evaluation
* UI
* integrations
* exports/reports

The foundation must remain **domain-agnostic**. Domain-specific schemas, prompts, scoring rules, workflows, or business logic belong in the vertical layer unless a capability is proven to be genuinely reusable.

## Decision rule

For every proposed change, ask:

1. Does this strengthen the reusable foundation?
2. Is it genuinely required by current requirements?
3. Will it remain useful across future verticals?
4. Does it preserve clean boundaries and provider independence?
5. Does it improve reliability, quality, performance, deployment, or maintainability?
6. Is this a foundation capability or merely a one-off feature?
7. Does this work cleanly on both Windows local and Linux server without platform-specific hacks in the core?
8. Does this increase deployment complexity for non-technical Windows users?
9. Does this materially improve customer time-to-value?

Prefer changes that strengthen the foundation and the current product.

Avoid adding domain-specific behavior to the foundation unless the abstraction is genuinely reusable.

## Avoid overengineering

Do NOT turn the foundation into a generic framework prematurely.

Do not add:

* speculative abstractions
* plugin systems without a real requirement
* unnecessary interfaces
* unnecessary microservices
* configuration explosion
* multiple retrieval mechanisms without benchmark evidence
* features solely because they may be useful someday

The foundation should remain **small, reliable, composable, and easy to verticalize**.

## Quality bar

The foundation is intended for real commercial use.

Optimize for:

**quality × speed × reliability × privacy × deployment simplicity × maintainability**

Do not optimize for feature count.

A technically impressive feature that makes deployment harder or reliability worse is not an improvement.

## Current priority order

1. Reliability and correct citations
2. Persian + English retrieval quality
3. Practical Windows deployment and ingestion speed
4. Customer time-to-value
5. Simple local deployment
6. Clean modular boundaries for future extension
7. Feature expansion

## Current execution focus

### Definition of Done for the first stable version

A non-technical user can:

1. Install the product on Windows with minimal friction.
2. Ingest real Persian and English PDF/DOCX documents.
3. Ask questions through a clean browser UI.
4. Receive reliable, citation-backed answers.
5. See the system refuse or clearly indicate when context is insufficient.

This is the success criterion for the first stable release.

### Immediate goal

Ship a high-quality, stable, demoable first version of the commercial product.

Prioritize now:

* reliable citations
* strong Persian + English retrieval quality
* practical Windows deployment and ingestion speed
* clean, convincing demo experience
* production stability of the core flow

Explicitly de-prioritize for now:

* perfect or complete foundation abstraction
* future vertical features
* advanced retrieval techniques without clear current benefit
* full public API, authentication, multi-tenancy, or enterprise hardening beyond what the first version needs

Do not delay the first sellable version in pursuit of a perfect foundation.
Build the smallest strong foundation that supports the current product well and remains easy to extend later.
The North Star remains valid, but execution priority is shipping a solid first version first.

## Productization principle

The core must support two deployment profiles without changing business logic.

### Local / Windows

* simple installer or setup script
* local services
* browser UI
* local embeddings when practical
* optional hosted LLM
* minimal technical knowledge required

### Server / Linux

* server deployment
* browser-based clients
* configurable embedding/LLM backends
* suitable as a future deployment profile for larger organizations and multi-user vertical products

The core architecture must remain shared.
Platform-specific details belong only in thin deployment layers or scripts, not in core business logic.

## Future API exposure

The foundation should be structured so a clean API layer can later be added without rewriting core logic.

This means:

* Core capabilities such as ingestion, retrieval, grounding, and citations should remain callable as services/modules rather than being tightly bound to the browser UI.
* Business logic must not live inside UI components.
* A future REST or similar API should be able to reuse the same foundation services used by the local/browser interface.

Do not implement a full public API now.
Do not add authentication, multi-tenancy, or API-specific infrastructure unless explicitly required.
Only preserve clean internal boundaries so future API exposure remains low-cost and non-disruptive.

## Verticalization principle

When a future client or project arrives, do not fork the foundation unnecessarily.

Instead:

```text
Foundation
   +
Domain layer
   +
Workflow layer
   +
Client-specific UI / integrations
```

Only modify the foundation when the requested capability is truly generic and reusable.

## AI provider principle

The foundation must not be coupled to one LLM or embedding provider.

Providers and models must remain configurable behind stable interfaces.

However, the customer-facing product should not expose unnecessary technical complexity.

The embedding layer may support both local and hosted execution profiles. Local embedding should remain the privacy-first default when practical, while hosted embedding may be used as an optional fast-ingestion path for large workloads or constrained hardware.

Internally:

```text
Provider abstraction
```

Externally:

```text
Simple curated configuration
```

## Retrieval principle

Retrieval quality is critical, but do not hard-code "Hybrid RAG" as a goal by itself.

The actual goal is:

> **High-quality retrieval for Persian, English, and mixed-language enterprise documents.**

Use semantic, lexical, hybrid, reranking, or other techniques only when evaluation demonstrates that they materially improve the product.

## Evaluation principle

Major architectural decisions must be evidence-driven.

Preferred flow:

```text
Hypothesis
→ benchmark
→ failure analysis
→ decision
→ minimal implementation
→ regression check
→ documentation
```

Do not introduce complexity based only on industry trends or theoretical advantages.

Any change to retrieval, chunking, or citation behavior should ideally include a small regression check against known Persian/English sample documents.

## Current product objective

The current commercial product is:

**A production-grade Windows-first private knowledge assistant for Persian and English documents, with simple deployment, high retrieval quality, reliable citations, and practical ingestion speed.**

It also serves as the reference implementation of the reusable foundation for future vertical AI knowledge products.

## What I expect from you

When implementing, refactoring, benchmarking, or architecting:

* evaluate against this strategy
* explicitly flag one-off coupling
* distinguish foundation work from vertical-specific work
* prefer reusable capabilities
* avoid unnecessary abstractions
* protect existing quality and reliability
* do not sacrifice production usability for architectural elegance
* always consider dual deployment (Windows-local and Linux-server)
* keep the core platform-agnostic and push platform-specific details to thin deployment layers

If a requested change conflicts with this strategy, explain the trade-off before implementing it.

This strategy does NOT override concrete project requirements or safety constraints. It is the strategic direction for how the foundation evolves.

