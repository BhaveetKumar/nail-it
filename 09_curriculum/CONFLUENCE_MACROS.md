---
# Auto-generated front matter
Title: Confluence Macros
LastUpdated: 2025-11-06T20:45:58.405508
Tags: []
Status: draft
---

# Confluence Macros for Master Engineer Curriculum

## Page Structure Macros

### Table of Contents
```confluence
{toc:printable=true|style=square|maxLevel=3|indent=20px|minLevel=1|exclude=[1//2]|type=list|outline=clear|include=.*}
```

### Info Panel
```confluence
{info:title=Master Engineer Curriculum}
A comprehensive learning path from fundamentals to distinguished engineer level with 550+ code examples and 100+ visual diagrams.
{info}
```

### Warning Panel
```confluence
{warning:title=Learning Path}
Start with Phase 0 fundamentals before progressing to advanced topics. Each phase builds upon the previous one.
{warning}
```

### Note Panel
```confluence
{note:title=Prerequisites}
- Basic programming knowledge
- Access to development environment
- Git for version control
{note}
```

## Code Block Macros

### Golang Code
```confluence
{code:language=go|title=Golang Implementation|linenumbers=true|collapse=false}
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
{code}
```

### Node.js Code
```confluence
{code:language=javascript|title=Node.js Implementation|linenumbers=true|collapse=false}
const express = require('express');
const app = express();

app.get('/', (req, res) => {
    res.send('Hello, World!');
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
{code}
```

### Mermaid Diagram
```confluence
{code:language=mermaid|title=System Architecture|linenumbers=false|collapse=false}
graph TD
    A[Client] --> B[API Gateway]
    B --> C[Microservices]
    C --> D[Database]
{code}
```

## Table Macros

### Curriculum Statistics Table
```confluence
{table}
| Phase | Modules | Files | Code Examples | Diagrams |
| Phase 0 | 4 | 8 | 120+ | 20+ |
| Phase 1 | 6 | 18 | 200+ | 40+ |
| Phase 2 | 6 | 10 | 150+ | 30+ |
| Phase 3 | 5 | 5 | 80+ | 15+ |
| **Total** | **21** | **41** | **550+** | **105+** |
{table}
```

### Learning Path Table
```confluence
{table:title=Learning Path|sortable=true}
| Phase | Duration | Focus | Prerequisites |
| Phase 0 | 8-12 weeks | Fundamentals | None |
| Phase 1 | 12-16 weeks | Intermediate | Phase 0 |
| Phase 2 | 16-20 weeks | Advanced | Phase 1 |
| Phase 3 | 20-24 weeks | Expert | Phase 2 |
{table}
```

## Status Macros

### Progress Indicators
```confluence
{status:colour=Green|title=Complete}
Phase 0: Fundamentals
{status}

{status:colour=Green|title=Complete}
Phase 1: Intermediate
{status}

{status:colour=Green|title=Complete}
Phase 2: Advanced
{status}

{status:colour=Green|title=Complete}
Phase 3: Expert
{status}
```

### Feature Status
```confluence
{panel:title=Implementation Status|borderStyle=solid|borderColor=#ccc|titleBGColor=#f7d6c1|bgColor=#ffffce}
✅ 35+ comprehensive implementation files
✅ 500+ production-ready code examples
✅ 100+ Mermaid diagrams
✅ Complete coverage of all phases
✅ Cross-linked learning paths
{panel}
```

## Navigation Macros

### Page Tree
```confluence
{pagetree:root=@self|startDepth=1|excerpt=true|searchBox=true|expandCollapseAll=true}
```

### Children Display
```confluence
{children:all=true|style=h4|excerpt=true|sort=creation|reverse=true}
```

## Content Macros

### Expand/Collapse Sections
```confluence
{expand:title=Phase 0: Fundamentals|icon=plus}
Complete coverage of mathematical foundations, programming basics, computer science fundamentals, and software engineering practices.
{expand}
```

### Column Layout
```confluence
{column:width=50%}
**Phase 0: Fundamentals**
- Mathematics
- Programming
- CS Basics
- Software Engineering
{column}

{column:width=50%}
**Phase 1: Intermediate**
- Advanced DSA
- OS Deep Dive
- Database Systems
- Web Development
{column}
```

### Quote Block
```confluence
{quote}
The Master Engineer Curriculum provides everything needed for engineers to progress from fundamentals to distinguished engineer level with practical, hands-on learning experiences.
{quote}
```

## Link Macros

### External Links
```confluence
[Golang Documentation|https://golang.org/doc/|target=_blank]
[Node.js Documentation|https://nodejs.org/docs/|target=_blank]
[Mermaid Documentation|https://mermaid-js.github.io/mermaid/|target=_blank]
```

### Internal Page Links
```confluence
[Phase 0 Fundamentals|phase0_fundamentals/README.md]
[Phase 1 Intermediate|phase1_intermediate/README.md]
[Phase 2 Advanced|phase2_advanced/README.md]
[Phase 3 Expert|phase3_expert/README.md]
```

## Image Macros

### Mermaid Diagram as Image
```confluence
{image:src=path/to/mermaid-diagram.png|alt=System Architecture Diagram|width=800|height=600}
```

### Logo/Banner
```confluence
{image:src=path/to/curriculum-logo.png|alt=Master Engineer Curriculum|width=400|height=200|align=center}
```

## Custom Macros for Smart Publisher

### Curriculum Overview Card
```confluence
{card:title=Master Engineer Curriculum|icon=graduation-cap|bgColor=#f0f8ff}
**Complete Learning Path**: From fundamentals to expert level
**550+ Code Examples**: Production-ready Golang & Node.js
**100+ Visual Diagrams**: Mermaid diagrams for complex concepts
**21 Modules**: Comprehensive coverage across 4 phases
{card}
```

### Phase Summary Cards
```confluence
{card:title=Phase 0: Fundamentals|icon=book|bgColor=#e8f5e8}
- Mathematics (4 modules)
- Programming (3 modules)
- CS Basics (4 modules)
- Software Engineering (4 modules)
{card}

{card:title=Phase 1: Intermediate|icon=rocket|bgColor=#fff3cd}
- Advanced DSA (5 modules)
- OS Deep Dive (5 modules)
- Database Systems (4 modules)
- Web Development (3 modules)
- API Design (3 modules)
- System Design Basics (2 modules)
{card}

{card:title=Phase 2: Advanced|icon=star|bgColor=#d1ecf1}
- Advanced Algorithms (3 modules)
- Cloud Architecture (2 modules)
- Machine Learning (2 modules)
- Performance Engineering (2 modules)
- Security Engineering (2 modules)
- Distributed Systems (4 modules)
{card}

{card:title=Phase 3: Expert|icon=crown|bgColor=#f8d7da}
- Technical Leadership (2 modules)
- Architecture Design (2 modules)
- Innovation Research (1 module)
- Mentoring Coaching (1 module)
- Strategic Planning (1 module)
{card}
```

## Usage Instructions

1. **Copy the macros** from this file
2. **Paste into Confluence** using the Smart Publisher extension
3. **Customize content** as needed for your specific Confluence instance
4. **Test rendering** to ensure all macros display correctly
5. **Publish** the complete curriculum to your engineering page

## Notes for Smart Publisher

- Use `{code}` blocks for all code examples
- Use `{table}` for structured data
- Use `{panel}` and `{card}` for visual organization
- Use `{status}` for progress indicators
- Use `{expand}` for collapsible content
- Use `{image}` for diagrams and visual content
