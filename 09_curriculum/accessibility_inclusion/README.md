# Accessibility & Inclusion

## Table of Contents

1. [Overview](#overview)
2. [Accessibility Standards](#accessibility-standards)
3. [Inclusive Design](#inclusive-design)
4. [Assistive Technologies](#assistive-technologies)
5. [Multilingual Support](#multilingual-support)
6. [Cultural Sensitivity](#cultural-sensitivity)
7. [Follow-up Questions](#follow-up-questions)
8. [Sources](#sources)

## Overview

### Learning Objectives

- Ensure the Master Engineer Curriculum is accessible to all learners
- Implement inclusive design principles
- Support diverse learning needs and preferences
- Create an inclusive learning environment

### What is Accessibility & Inclusion?

Accessibility and inclusion involve designing the Master Engineer Curriculum to be usable by people with diverse abilities, backgrounds, and learning preferences, ensuring equal access to learning opportunities.

## Accessibility Standards

### 1. WCAG Compliance

#### Web Content Accessibility Guidelines
```html
<!-- Accessible HTML Structure -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Master Engineer Curriculum - Lesson</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header role="banner">
        <nav role="navigation" aria-label="Main navigation">
            <ul>
                <li><a href="#main-content" class="skip-link">Skip to main content</a></li>
                <li><a href="/lessons" aria-current="page">Lessons</a></li>
                <li><a href="/progress">Progress</a></li>
                <li><a href="/profile">Profile</a></li>
            </ul>
        </nav>
    </header>

    <main id="main-content" role="main">
        <article>
            <header>
                <h1>Linear Algebra Fundamentals</h1>
                <p class="lesson-meta">
                    <span class="duration">Duration: 2 hours</span>
                    <span class="difficulty">Difficulty: Intermediate</span>
                    <span class="prerequisites">Prerequisites: Basic Mathematics</span>
                </p>
            </header>

            <section aria-labelledby="overview-heading">
                <h2 id="overview-heading">Overview</h2>
                <p>Linear algebra is a fundamental branch of mathematics that deals with vector spaces and linear mappings between these spaces.</p>
            </section>

            <section aria-labelledby="objectives-heading">
                <h2 id="objectives-heading">Learning Objectives</h2>
                <ul>
                    <li>Understand vector operations and properties</li>
                    <li>Learn matrix multiplication and transformations</li>
                    <li>Apply linear algebra concepts to real-world problems</li>
                </ul>
            </section>

            <section aria-labelledby="content-heading">
                <h2 id="content-heading">Content</h2>
                <div class="code-example" role="region" aria-labelledby="code-heading">
                    <h3 id="code-heading">Go Implementation</h3>
                    <pre><code>package main

import "fmt"

// Vector represents a mathematical vector
type Vector struct {
    Components []float64
}

// Add performs vector addition
func (v Vector) Add(other Vector) Vector {
    if len(v.Components) != len(other.Components) {
        panic("Vector dimensions must match")
    }
    
    result := make([]float64, len(v.Components))
    for i, component := range v.Components {
        result[i] = component + other.Components[i]
    }
    
    return Vector{Components: result}
}

func main() {
    v1 := Vector{Components: []float64{1, 2, 3}}
    v2 := Vector{Components: []float64{4, 5, 6}}
    
    result := v1.Add(v2)
    fmt.Printf("Vector addition result: %v\n", result.Components)
}</code></pre>
                </div>
            </section>
        </article>
    </main>

    <footer role="contentinfo">
        <p>&copy; 2024 Master Engineer Curriculum. All rights reserved.</p>
    </footer>
</body>
</html>
```

### 2. ARIA Implementation

#### Accessible Rich Internet Applications
```javascript
// Accessible JavaScript Components
class AccessibleLesson {
    constructor(container) {
        this.container = container;
        this.currentStep = 0;
        this.steps = container.querySelectorAll('.lesson-step');
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupKeyboardSupport();
        this.announceProgress();
    }

    setupNavigation() {
        const prevButton = this.container.querySelector('.prev-button');
        const nextButton = this.container.querySelector('.next-button');
        
        if (prevButton) {
            prevButton.addEventListener('click', () => this.previousStep());
            prevButton.setAttribute('aria-label', 'Go to previous lesson step');
        }
        
        if (nextButton) {
            nextButton.addEventListener('click', () => this.nextStep());
            nextButton.setAttribute('aria-label', 'Go to next lesson step');
        }
    }

    setupKeyboardSupport() {
        this.container.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.previousStep();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.nextStep();
                    break;
                case 'Home':
                    e.preventDefault();
                    this.goToStep(0);
                    break;
                case 'End':
                    e.preventDefault();
                    this.goToStep(this.steps.length - 1);
                    break;
            }
        });
    }

    nextStep() {
        if (this.currentStep < this.steps.length - 1) {
            this.goToStep(this.currentStep + 1);
        }
    }

    previousStep() {
        if (this.currentStep > 0) {
            this.goToStep(this.currentStep - 1);
        }
    }

    goToStep(stepIndex) {
        // Hide current step
        this.steps[this.currentStep].setAttribute('aria-hidden', 'true');
        this.steps[this.currentStep].classList.remove('active');
        
        // Show new step
        this.currentStep = stepIndex;
        this.steps[this.currentStep].setAttribute('aria-hidden', 'false');
        this.steps[this.currentStep].classList.add('active');
        
        // Update progress indicator
        this.updateProgressIndicator();
        
        // Announce progress
        this.announceProgress();
        
        // Focus on new step
        this.steps[this.currentStep].focus();
    }

    updateProgressIndicator() {
        const progressBar = this.container.querySelector('.progress-bar');
        if (progressBar) {
            const progress = ((this.currentStep + 1) / this.steps.length) * 100;
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
        }
    }

    announceProgress() {
        const announcement = `Step ${this.currentStep + 1} of ${this.steps.length}`;
        this.announceToScreenReader(announcement);
    }

    announceToScreenReader(message) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'polite');
        announcement.setAttribute('aria-atomic', 'true');
        announcement.className = 'sr-only';
        announcement.textContent = message;
        
        document.body.appendChild(announcement);
        
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }
}

// Initialize accessible lesson
document.addEventListener('DOMContentLoaded', () => {
    const lessonContainer = document.querySelector('.lesson-container');
    if (lessonContainer) {
        new AccessibleLesson(lessonContainer);
    }
});
```

## Inclusive Design

### 1. Universal Design Principles

#### Design for All
```css
/* Inclusive CSS Design */
:root {
    /* High contrast colors */
    --primary-color: #0066cc;
    --primary-contrast: #ffffff;
    --secondary-color: #f0f0f0;
    --text-color: #333333;
    --text-contrast: #ffffff;
    --error-color: #cc0000;
    --success-color: #006600;
    
    /* Accessible font sizes */
    --font-size-small: 0.875rem;
    --font-size-base: 1rem;
    --font-size-large: 1.125rem;
    --font-size-xl: 1.25rem;
    
    /* Accessible spacing */
    --spacing-xs: 0.25rem;
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --spacing-xl: 2rem;
    
    /* Accessible line heights */
    --line-height-tight: 1.25;
    --line-height-normal: 1.5;
    --line-height-relaxed: 1.75;
}

/* Base styles for all users */
* {
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: var(--font-size-base);
    line-height: var(--line-height-normal);
    color: var(--text-color);
    background-color: #ffffff;
}

/* Focus indicators */
*:focus {
    outline: 2px solid var(--primary-color);
    outline-offset: 2px;
}

/* Skip links for keyboard navigation */
.skip-link {
    position: absolute;
    top: -40px;
    left: 6px;
    background: var(--primary-color);
    color: var(--primary-contrast);
    padding: var(--spacing-sm) var(--spacing-md);
    text-decoration: none;
    border-radius: 4px;
    z-index: 1000;
}

.skip-link:focus {
    top: 6px;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
    :root {
        --primary-color: #0000ff;
        --text-color: #000000;
        --background-color: #ffffff;
    }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    :root {
        --text-color: #ffffff;
        --background-color: #1a1a1a;
        --secondary-color: #333333;
    }
}

/* Screen reader only content */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* Accessible buttons */
.btn {
    display: inline-block;
    padding: var(--spacing-sm) var(--spacing-md);
    background-color: var(--primary-color);
    color: var(--primary-contrast);
    border: none;
    border-radius: 4px;
    font-size: var(--font-size-base);
    line-height: var(--line-height-normal);
    cursor: pointer;
    text-decoration: none;
    transition: background-color 0.2s ease;
}

.btn:hover {
    background-color: #0052a3;
}

.btn:focus {
    outline: 2px solid var(--primary-color);
    outline-offset: 2px;
}

.btn:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
}

/* Accessible form elements */
.form-group {
    margin-bottom: var(--spacing-md);
}

.form-label {
    display: block;
    margin-bottom: var(--spacing-xs);
    font-weight: 600;
    color: var(--text-color);
}

.form-input {
    width: 100%;
    padding: var(--spacing-sm);
    border: 2px solid #cccccc;
    border-radius: 4px;
    font-size: var(--font-size-base);
    line-height: var(--line-height-normal);
}

.form-input:focus {
    border-color: var(--primary-color);
    outline: none;
}

.form-input:invalid {
    border-color: var(--error-color);
}

/* Accessible tables */
.table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: var(--spacing-md);
}

.table th,
.table td {
    padding: var(--spacing-sm);
    text-align: left;
    border-bottom: 1px solid #cccccc;
}

.table th {
    background-color: var(--secondary-color);
    font-weight: 600;
}

/* Accessible code blocks */
.code-block {
    background-color: #f8f8f8;
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: var(--spacing-md);
    overflow-x: auto;
    font-family: 'Courier New', monospace;
    font-size: var(--font-size-small);
    line-height: var(--line-height-normal);
}

/* Accessible progress indicators */
.progress-bar {
    width: 100%;
    height: 20px;
    background-color: var(--secondary-color);
    border-radius: 10px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background-color: var(--primary-color);
    transition: width 0.3s ease;
}

/* Accessible navigation */
.nav-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.nav-item {
    display: inline-block;
    margin-right: var(--spacing-md);
}

.nav-link {
    display: block;
    padding: var(--spacing-sm) var(--spacing-md);
    color: var(--text-color);
    text-decoration: none;
    border-radius: 4px;
}

.nav-link:hover {
    background-color: var(--secondary-color);
}

.nav-link:focus {
    outline: 2px solid var(--primary-color);
    outline-offset: 2px;
}

.nav-link[aria-current="page"] {
    background-color: var(--primary-color);
    color: var(--primary-contrast);
}
```

### 2. Responsive Design

#### Mobile-First Approach
```css
/* Mobile-first responsive design */
.container {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 var(--spacing-md);
}

/* Small screens (mobile) */
@media (max-width: 767px) {
    .lesson-content {
        padding: var(--spacing-md);
    }
    
    .code-block {
        font-size: 0.75rem;
        padding: var(--spacing-sm);
    }
    
    .btn {
        width: 100%;
        margin-bottom: var(--spacing-sm);
    }
    
    .table {
        font-size: var(--font-size-small);
    }
    
    .table th,
    .table td {
        padding: var(--spacing-xs);
    }
}

/* Medium screens (tablet) */
@media (min-width: 768px) and (max-width: 1023px) {
    .lesson-content {
        padding: var(--spacing-lg);
    }
    
    .code-block {
        font-size: var(--font-size-small);
    }
    
    .btn {
        width: auto;
        margin-right: var(--spacing-sm);
    }
}

/* Large screens (desktop) */
@media (min-width: 1024px) {
    .lesson-content {
        padding: var(--spacing-xl);
    }
    
    .code-block {
        font-size: var(--font-size-base);
    }
    
    .btn {
        width: auto;
        margin-right: var(--spacing-md);
    }
}

/* Extra large screens */
@media (min-width: 1200px) {
    .container {
        max-width: 1400px;
    }
    
    .lesson-content {
        padding: var(--spacing-xl) var(--spacing-xl);
    }
}
```

## Assistive Technologies

### 1. Screen Reader Support

#### Screen Reader Optimization
```html
<!-- Screen reader friendly content -->
<article role="main" aria-labelledby="lesson-title">
    <header>
        <h1 id="lesson-title">Data Structures and Algorithms</h1>
        <p class="lesson-description" aria-describedby="lesson-title">
            Learn fundamental data structures and algorithms used in software development.
        </p>
    </header>

    <nav aria-label="Lesson navigation">
        <h2>Table of Contents</h2>
        <ol>
            <li><a href="#overview" aria-describedby="overview-desc">Overview</a></li>
            <li><a href="#arrays" aria-describedby="arrays-desc">Arrays</a></li>
            <li><a href="#linked-lists" aria-describedby="linked-lists-desc">Linked Lists</a></li>
            <li><a href="#trees" aria-describedby="trees-desc">Trees</a></li>
            <li><a href="#graphs" aria-describedby="graphs-desc">Graphs</a></li>
        </ol>
    </nav>

    <section id="overview" aria-labelledby="overview-heading">
        <h2 id="overview-heading">Overview</h2>
        <p id="overview-desc">
            Data structures are ways of organizing and storing data in a computer so that it can be accessed and modified efficiently.
        </p>
        
        <div class="key-concepts" role="region" aria-labelledby="key-concepts-heading">
            <h3 id="key-concepts-heading">Key Concepts</h3>
            <ul>
                <li>Time complexity analysis</li>
                <li>Space complexity analysis</li>
                <li>Algorithm design patterns</li>
                <li>Data structure selection</li>
            </ul>
        </div>
    </section>

    <section id="arrays" aria-labelledby="arrays-heading">
        <h2 id="arrays-heading">Arrays</h2>
        <p id="arrays-desc">
            Arrays are a collection of elements stored in contiguous memory locations.
        </p>
        
        <div class="code-example" role="region" aria-labelledby="array-code-heading">
            <h3 id="array-code-heading">Array Implementation in Go</h3>
            <pre><code>package main

import "fmt"

func main() {
    // Declare and initialize an array
    numbers := [5]int{1, 2, 3, 4, 5}
    
    // Access elements
    fmt.Printf("First element: %d\n", numbers[0])
    fmt.Printf("Last element: %d\n", numbers[4])
    
    // Iterate through array
    for i, value := range numbers {
        fmt.Printf("Index %d: %d\n", i, value)
    }
}</code></pre>
        </div>
        
        <div class="interactive-example" role="region" aria-labelledby="array-example-heading">
            <h3 id="array-example-heading">Interactive Array Example</h3>
            <p>Try creating an array with different values:</p>
            <form>
                <label for="array-size">Array size:</label>
                <input type="number" id="array-size" min="1" max="10" value="5" aria-describedby="array-size-help">
                <div id="array-size-help" class="help-text">Enter a number between 1 and 10</div>
                
                <button type="button" onclick="createArray()">Create Array</button>
            </form>
            
            <div id="array-output" aria-live="polite" aria-atomic="true">
                <!-- Array will be displayed here -->
            </div>
        </div>
    </section>
</article>
```

### 2. Keyboard Navigation

#### Keyboard Accessibility
```javascript
// Keyboard navigation support
class KeyboardNavigation {
    constructor(container) {
        this.container = container;
        this.focusableElements = this.getFocusableElements();
        this.currentIndex = 0;
        this.init();
    }

    init() {
        this.container.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.setupFocusManagement();
    }

    getFocusableElements() {
        const selector = 'a[href], button, input, textarea, select, [tabindex]:not([tabindex="-1"])';
        return Array.from(this.container.querySelectorAll(selector));
    }

    handleKeyDown(e) {
        switch(e.key) {
            case 'Tab':
                this.handleTab(e);
                break;
            case 'ArrowDown':
                e.preventDefault();
                this.moveFocus(1);
                break;
            case 'ArrowUp':
                e.preventDefault();
                this.moveFocus(-1);
                break;
            case 'Home':
                e.preventDefault();
                this.moveFocusToFirst();
                break;
            case 'End':
                e.preventDefault();
                this.moveFocusToLast();
                break;
            case 'Enter':
            case ' ':
                this.activateElement(e);
                break;
        }
    }

    handleTab(e) {
        if (e.shiftKey) {
            this.moveFocus(-1);
        } else {
            this.moveFocus(1);
        }
    }

    moveFocus(direction) {
        this.currentIndex += direction;
        
        if (this.currentIndex < 0) {
            this.currentIndex = this.focusableElements.length - 1;
        } else if (this.currentIndex >= this.focusableElements.length) {
            this.currentIndex = 0;
        }
        
        this.focusableElements[this.currentIndex].focus();
    }

    moveFocusToFirst() {
        this.currentIndex = 0;
        this.focusableElements[this.currentIndex].focus();
    }

    moveFocusToLast() {
        this.currentIndex = this.focusableElements.length - 1;
        this.focusableElements[this.currentIndex].focus();
    }

    activateElement(e) {
        const element = e.target;
        
        if (element.tagName === 'A' || element.tagName === 'BUTTON') {
            element.click();
        } else if (element.type === 'checkbox' || element.type === 'radio') {
            element.checked = !element.checked;
        }
    }

    setupFocusManagement() {
        // Trap focus within modal dialogs
        const modals = this.container.querySelectorAll('[role="dialog"]');
        modals.forEach(modal => {
            modal.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    this.closeModal(modal);
                }
            });
        });
    }

    closeModal(modal) {
        modal.style.display = 'none';
        modal.setAttribute('aria-hidden', 'true');
        
        // Return focus to trigger element
        const trigger = document.querySelector(`[aria-controls="${modal.id}"]`);
        if (trigger) {
            trigger.focus();
        }
    }
}

// Initialize keyboard navigation
document.addEventListener('DOMContentLoaded', () => {
    const mainContainer = document.querySelector('main');
    if (mainContainer) {
        new KeyboardNavigation(mainContainer);
    }
});
```

## Multilingual Support

### 1. Internationalization

#### i18n Implementation
```go
// i18n/translations.go
package main

import (
    "encoding/json"
    "fmt"
    "os"
)

type Translation struct {
    Key    string `json:"key"`
    Value  string `json:"value"`
    Locale string `json:"locale"`
}

type TranslationManager struct {
    translations map[string]map[string]string
    currentLocale string
}

func NewTranslationManager() *TranslationManager {
    return &TranslationManager{
        translations: make(map[string]map[string]string),
        currentLocale: "en",
    }
}

func (tm *TranslationManager) LoadTranslations(locale string) error {
    filename := fmt.Sprintf("translations/%s.json", locale)
    data, err := os.ReadFile(filename)
    if err != nil {
        return fmt.Errorf("failed to load translations for %s: %w", locale, err)
    }
    
    var translations []Translation
    if err := json.Unmarshal(data, &translations); err != nil {
        return fmt.Errorf("failed to parse translations: %w", err)
    }
    
    tm.translations[locale] = make(map[string]string)
    for _, t := range translations {
        tm.translations[locale][t.Key] = t.Value
    }
    
    return nil
}

func (tm *TranslationManager) SetLocale(locale string) {
    tm.currentLocale = locale
}

func (tm *TranslationManager) Translate(key string) string {
    if translations, exists := tm.translations[tm.currentLocale]; exists {
        if value, exists := translations[key]; exists {
            return value
        }
    }
    
    // Fallback to English
    if translations, exists := tm.translations["en"]; exists {
        if value, exists := translations[key]; exists {
            return value
        }
    }
    
    return key // Return key if translation not found
}

// Translation files
// translations/en.json
{
    "lesson.title": "Lesson",
    "lesson.overview": "Overview",
    "lesson.objectives": "Learning Objectives",
    "lesson.content": "Content",
    "lesson.exercises": "Exercises",
    "lesson.quiz": "Quiz",
    "lesson.next": "Next Lesson",
    "lesson.previous": "Previous Lesson",
    "lesson.complete": "Mark as Complete",
    "progress.overall": "Overall Progress",
    "progress.completed": "Completed",
    "progress.remaining": "Remaining",
    "progress.time_spent": "Time Spent",
    "progress.streak": "Learning Streak",
    "assessment.score": "Score",
    "assessment.passed": "Passed",
    "assessment.failed": "Failed",
    "assessment.retry": "Retry",
    "navigation.home": "Home",
    "navigation.lessons": "Lessons",
    "navigation.progress": "Progress",
    "navigation.profile": "Profile",
    "navigation.settings": "Settings"
}

// translations/es.json
{
    "lesson.title": "Lección",
    "lesson.overview": "Resumen",
    "lesson.objectives": "Objetivos de Aprendizaje",
    "lesson.content": "Contenido",
    "lesson.exercises": "Ejercicios",
    "lesson.quiz": "Cuestionario",
    "lesson.next": "Siguiente Lección",
    "lesson.previous": "Lección Anterior",
    "lesson.complete": "Marcar como Completado",
    "progress.overall": "Progreso General",
    "progress.completed": "Completado",
    "progress.remaining": "Restante",
    "progress.time_spent": "Tiempo Invertido",
    "progress.streak": "Racha de Aprendizaje",
    "assessment.score": "Puntuación",
    "assessment.passed": "Aprobado",
    "assessment.failed": "Reprobado",
    "assessment.retry": "Reintentar",
    "navigation.home": "Inicio",
    "navigation.lessons": "Lecciones",
    "navigation.progress": "Progreso",
    "navigation.profile": "Perfil",
    "navigation.settings": "Configuración"
}

// translations/fr.json
{
    "lesson.title": "Leçon",
    "lesson.overview": "Aperçu",
    "lesson.objectives": "Objectifs d'Apprentissage",
    "lesson.content": "Contenu",
    "lesson.exercises": "Exercices",
    "lesson.quiz": "Quiz",
    "lesson.next": "Leçon Suivante",
    "lesson.previous": "Leçon Précédente",
    "lesson.complete": "Marquer comme Terminé",
    "progress.overall": "Progrès Global",
    "progress.completed": "Terminé",
    "progress.remaining": "Restant",
    "progress.time_spent": "Temps Passé",
    "progress.streak": "Série d'Apprentissage",
    "assessment.score": "Score",
    "assessment.passed": "Réussi",
    "assessment.failed": "Échoué",
    "assessment.retry": "Réessayer",
    "navigation.home": "Accueil",
    "navigation.lessons": "Leçons",
    "navigation.progress": "Progrès",
    "navigation.profile": "Profil",
    "navigation.settings": "Paramètres"
}
```

### 2. RTL Support

#### Right-to-Left Language Support
```css
/* RTL Support */
[dir="rtl"] {
    text-align: right;
}

[dir="rtl"] .container {
    direction: rtl;
}

[dir="rtl"] .nav-list {
    text-align: right;
}

[dir="rtl"] .form-input {
    text-align: right;
}

[dir="rtl"] .code-block {
    direction: ltr;
    text-align: left;
}

[dir="rtl"] .table {
    direction: rtl;
}

[dir="rtl"] .table th,
[dir="rtl"] .table td {
    text-align: right;
}

/* RTL specific adjustments */
[dir="rtl"] .btn {
    margin-right: 0;
    margin-left: var(--spacing-sm);
}

[dir="rtl"] .skip-link {
    right: 6px;
    left: auto;
}

[dir="rtl"] .progress-bar {
    direction: rtl;
}

/* Mixed content support */
.mixed-content {
    unicode-bidi: embed;
}

.mixed-content .ltr {
    direction: ltr;
    unicode-bidi: embed;
}

.mixed-content .rtl {
    direction: rtl;
    unicode-bidi: embed;
}
```

## Cultural Sensitivity

### 1. Inclusive Content

#### Cultural Considerations
```go
// content/cultural_sensitivity.go
package main

import (
    "context"
    "fmt"
)

type CulturalSensitivity struct {
    contentRepo ContentRepository
    userRepo    UserRepository
}

type CulturalContext struct {
    Region      string
    Language    string
    TimeZone    string
    Currency    string
    DateFormat  string
    NumberFormat string
}

func (cs *CulturalSensitivity) AdaptContent(ctx context.Context, contentID string, userID string) (*AdaptedContent, error) {
    // Get user's cultural context
    user, err := cs.userRepo.GetByID(ctx, userID)
    if err != nil {
        return nil, fmt.Errorf("failed to get user: %w", err)
    }
    
    culturalContext := cs.getCulturalContext(user)
    
    // Get original content
    content, err := cs.contentRepo.GetByID(ctx, contentID)
    if err != nil {
        return nil, fmt.Errorf("failed to get content: %w", err)
    }
    
    // Adapt content based on cultural context
    adaptedContent := &AdaptedContent{
        ID:      content.ID,
        Title:   cs.adaptTitle(content.Title, culturalContext),
        Content: cs.adaptContent(content.Content, culturalContext),
        Examples: cs.adaptExamples(content.Examples, culturalContext),
        References: cs.adaptReferences(content.References, culturalContext),
    }
    
    return adaptedContent, nil
}

func (cs *CulturalSensitivity) getCulturalContext(user *User) *CulturalContext {
    return &CulturalContext{
        Region:       user.Region,
        Language:     user.Language,
        TimeZone:     user.TimeZone,
        Currency:     user.Currency,
        DateFormat:   user.DateFormat,
        NumberFormat: user.NumberFormat,
    }
}

func (cs *CulturalSensitivity) adaptTitle(title string, context *CulturalContext) string {
    // Adapt title based on cultural context
    switch context.Region {
    case "US":
        return title
    case "UK":
        return cs.adaptToBritishEnglish(title)
    case "AU":
        return cs.adaptToAustralianEnglish(title)
    default:
        return title
    }
}

func (cs *CulturalSensitivity) adaptContent(content string, context *CulturalContext) string {
    // Adapt content based on cultural context
    adapted := content
    
    // Adapt currency references
    adapted = cs.adaptCurrency(adapted, context.Currency)
    
    // Adapt date formats
    adapted = cs.adaptDateFormat(adapted, context.DateFormat)
    
    // Adapt number formats
    adapted = cs.adaptNumberFormat(adapted, context.NumberFormat)
    
    // Adapt cultural references
    adapted = cs.adaptCulturalReferences(adapted, context.Region)
    
    return adapted
}

func (cs *CulturalSensitivity) adaptExamples(examples []Example, context *CulturalContext) []Example {
    adapted := make([]Example, len(examples))
    
    for i, example := range examples {
        adapted[i] = Example{
            ID:          example.ID,
            Title:       cs.adaptTitle(example.Title, context),
            Description: cs.adaptContent(example.Description, context),
            Code:        example.Code, // Code examples are generally culture-neutral
            Output:      cs.adaptOutput(example.Output, context),
        }
    }
    
    return adapted
}

func (cs *CulturalSensitivity) adaptReferences(references []Reference, context *CulturalContext) []Reference {
    adapted := make([]Reference, 0)
    
    for _, ref := range references {
        // Filter references based on cultural relevance
        if cs.isRelevantToContext(ref, context) {
            adapted = append(adapted, Reference{
                ID:          ref.ID,
                Title:       cs.adaptTitle(ref.Title, context),
                URL:         ref.URL,
                Description: cs.adaptContent(ref.Description, context),
                Language:    cs.adaptLanguage(ref.Language, context.Language),
            })
        }
    }
    
    return adapted
}

func (cs *CulturalSensitivity) adaptCurrency(content string, currency string) string {
    // Replace currency symbols and formats based on region
    switch currency {
    case "USD":
        return content
    case "EUR":
        return cs.replaceCurrencySymbols(content, "$", "€")
    case "GBP":
        return cs.replaceCurrencySymbols(content, "$", "£")
    case "JPY":
        return cs.replaceCurrencySymbols(content, "$", "¥")
    default:
        return content
    }
}

func (cs *CulturalSensitivity) adaptDateFormat(content string, dateFormat string) string {
    // Adapt date formats based on region
    switch dateFormat {
    case "MM/DD/YYYY":
        return content
    case "DD/MM/YYYY":
        return cs.convertDateFormat(content, "MM/DD/YYYY", "DD/MM/YYYY")
    case "YYYY-MM-DD":
        return cs.convertDateFormat(content, "MM/DD/YYYY", "YYYY-MM-DD")
    default:
        return content
    }
}

func (cs *CulturalSensitivity) adaptNumberFormat(content string, numberFormat string) string {
    // Adapt number formats based on region
    switch numberFormat {
    case "US":
        return content
    case "EU":
        return cs.convertNumberFormat(content, ".", ",")
    default:
        return content
    }
}

func (cs *CulturalSensitivity) adaptCulturalReferences(content string, region string) string {
    // Adapt cultural references based on region
    switch region {
    case "US":
        return content
    case "UK":
        return cs.adaptToBritishContext(content)
    case "AU":
        return cs.adaptToAustralianContext(content)
    case "CA":
        return cs.adaptToCanadianContext(content)
    default:
        return content
    }
}

func (cs *CulturalSensitivity) isRelevantToContext(ref *Reference, context *CulturalContext) bool {
    // Check if reference is relevant to user's cultural context
    if ref.Language != "" && ref.Language != context.Language {
        return false
    }
    
    if ref.Region != "" && ref.Region != context.Region {
        return false
    }
    
    return true
}
```

## Follow-up Questions

### 1. Accessibility
**Q: How do you ensure the curriculum is accessible to all learners?**
A: Follow WCAG guidelines, implement ARIA attributes, support keyboard navigation, provide screen reader compatibility, and test with assistive technologies.

### 2. Inclusion
**Q: What makes the curriculum inclusive for diverse learners?**
A: Use inclusive design principles, support multiple languages, adapt content culturally, provide various learning formats, and ensure equal access to all features.

### 3. Cultural Sensitivity
**Q: How do you handle cultural differences in content?**
A: Adapt examples, references, and formats based on user's cultural context, use inclusive language, and provide culturally relevant content.

## Sources

### Accessibility
- **WCAG**: [Web Content Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- **ARIA**: [Accessible Rich Internet Applications](https://www.w3.org/WAI/ARIA/apg/)
- **Section 508**: [Accessibility Standards](https://www.section508.gov/)

### Inclusion
- **Universal Design**: [Design for All](https://universaldesign.org/)
- **Inclusive Design**: [Microsoft Inclusive Design](https://www.microsoft.com/design/inclusive/)
- **Diversity & Inclusion**: [Tech Inclusion](https://techinclusion.co/)

### Multilingual
- **i18n**: [Internationalization](https://www.w3.org/International/)
- **RTL Support**: [Right-to-Left Languages](https://www.w3.org/International/questions/qa-html-dir/)
- **Translation**: [Localization Best Practices](https://www.w3.org/International/techniques/developing-specs/)

---

**Next**: [Analytics Insights](../../README.md) | **Previous**: [Community Contributions](../../README.md) | **Up**: [Accessibility Inclusion](README.md/)
