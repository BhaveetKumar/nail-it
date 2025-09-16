# Frontend Frameworks

## Overview

This module covers frontend framework concepts including React, Vue.js, Angular, state management, component lifecycle, and performance optimization. These concepts are essential for building modern, interactive web applications.

## Table of Contents

1. [React Fundamentals](#react-fundamentals)
2. [Vue.js Fundamentals](#vuejs-fundamentals)
3. [Angular Fundamentals](#angular-fundamentals)
4. [State Management](#state-management)
5. [Performance Optimization](#performance-optimization)
6. [Applications](#applications)
7. [Complexity Analysis](#complexity-analysis)
8. [Follow-up Questions](#follow-up-questions)

## React Fundamentals

### Theory

React is a JavaScript library for building user interfaces, particularly single-page applications. It uses a virtual DOM for efficient updates and follows a component-based architecture.

### React Component Implementation

#### Golang Implementation (Server-Side Rendering)

```go
package main

import (
    "fmt"
    "html/template"
    "net/http"
    "strconv"
    "time"
)

type ReactComponent struct {
    Name       string
    Props      map[string]interface{}
    State      map[string]interface{}
    Children   []*ReactComponent
    Lifecycle  []string
}

type ReactApp struct {
    Components map[string]*ReactComponent
    Routes     map[string]string
    State      map[string]interface{}
}

func NewReactApp() *ReactApp {
    return &ReactApp{
        Components: make(map[string]*ReactComponent),
        Routes:     make(map[string]string),
        State:      make(map[string]interface{}),
    }
}

func (ra *ReactApp) CreateComponent(name string, props map[string]interface{}) *ReactComponent {
    component := &ReactComponent{
        Name:      name,
        Props:     props,
        State:     make(map[string]interface{}),
        Children:  make([]*ReactComponent, 0),
        Lifecycle: make([]string, 0),
    }
    
    ra.Components[name] = component
    ra.addLifecycleEvent(component, "constructor")
    
    fmt.Printf("Created React component: %s\n", name)
    return component
}

func (ra *ReactApp) addLifecycleEvent(component *ReactComponent, event string) {
    component.Lifecycle = append(component.Lifecycle, event)
    fmt.Printf("Component %s: %s\n", component.Name, event)
}

func (ra *ReactApp) MountComponent(componentName, parentName string) {
    if component, exists := ra.Components[componentName]; exists {
        if parent, exists := ra.Components[parentName]; exists {
            parent.Children = append(parent.Children, component)
            ra.addLifecycleEvent(component, "componentDidMount")
        }
    }
}

func (ra *ReactApp) UpdateComponent(componentName string, newProps map[string]interface{}) {
    if component, exists := ra.Components[componentName]; exists {
        ra.addLifecycleEvent(component, "componentWillReceiveProps")
        
        // Update props
        for key, value := range newProps {
            component.Props[key] = value
        }
        
        ra.addLifecycleEvent(component, "componentDidUpdate")
    }
}

func (ra *ReactApp) UnmountComponent(componentName string) {
    if component, exists := ra.Components[componentName]; exists {
        ra.addLifecycleEvent(component, "componentWillUnmount")
        delete(ra.Components, componentName)
    }
}

func (ra *ReactApp) SetState(componentName string, newState map[string]interface{}) {
    if component, exists := ra.Components[componentName]; exists {
        ra.addLifecycleEvent(component, "componentWillUpdate")
        
        // Update state
        for key, value := range newState {
            component.State[key] = value
        }
        
        ra.addLifecycleEvent(component, "componentDidUpdate")
    }
}

func (ra *ReactApp) RenderComponent(componentName string) string {
    if component, exists := ra.Components[componentName]; exists {
        ra.addLifecycleEvent(component, "render")
        
        // Generate HTML for component
        html := fmt.Sprintf("<div class=\"%s\">", component.Name)
        
        // Render props
        for key, value := range component.Props {
            html += fmt.Sprintf(" %s=\"%v\"", key, value)
        }
        
        // Render state
        for key, value := range component.State {
            html += fmt.Sprintf(" data-%s=\"%v\"", key, value)
        }
        
        html += ">"
        
        // Render children
        for _, child := range component.Children {
            html += ra.RenderComponent(child.Name)
        }
        
        html += "</div>"
        return html
    }
    
    return ""
}

func (ra *ReactApp) AddRoute(path, componentName string) {
    ra.Routes[path] = componentName
    fmt.Printf("Added route: %s -> %s\n", path, componentName)
}

func (ra *ReactApp) HandleRequest(w http.ResponseWriter, r *http.Request) {
    path := r.URL.Path
    
    if componentName, exists := ra.Routes[path]; exists {
        html := ra.RenderComponent(componentName)
        
        // Serve the rendered component
        w.Header().Set("Content-Type", "text/html")
        fmt.Fprintf(w, `
<!DOCTYPE html>
<html>
<head>
    <title>React App</title>
    <style>
        .component { border: 1px solid #ccc; margin: 10px; padding: 10px; }
        .App { background-color: #f0f0f0; }
        .Header { background-color: #333; color: white; }
        .Content { background-color: white; }
    </style>
</head>
<body>
    %s
</body>
</html>`, html)
    } else {
        http.NotFound(w, r)
    }
}

func main() {
    app := NewReactApp()
    
    fmt.Println("React App Demo:")
    
    // Create components
    app.CreateComponent("App", map[string]interface{}{
        "title": "My React App",
    })
    
    app.CreateComponent("Header", map[string]interface{}{
        "text": "Welcome to React",
    })
    
    app.CreateComponent("Content", map[string]interface{}{
        "message": "Hello, World!",
    })
    
    // Mount components
    app.MountComponent("Header", "App")
    app.MountComponent("Content", "App")
    
    // Set state
    app.SetState("Content", map[string]interface{}{
        "count": 0,
    })
    
    // Update props
    app.UpdateComponent("Header", map[string]interface{}{
        "text": "Welcome to React - Updated!",
    })
    
    // Add routes
    app.AddRoute("/", "App")
    app.AddRoute("/home", "App")
    
    // Render component
    html := app.RenderComponent("App")
    fmt.Printf("Rendered HTML:\n%s\n", html)
    
    // Start HTTP server
    http.HandleFunc("/", app.HandleRequest)
    fmt.Println("Server starting on :8080")
    go http.ListenAndServe(":8080", nil)
    
    // Keep the program running
    time.Sleep(1 * time.Second)
}
```

## Vue.js Fundamentals

### Theory

Vue.js is a progressive JavaScript framework for building user interfaces. It features a template-based syntax, reactive data binding, and a component system similar to React.

### Vue.js Component Implementation

#### Golang Implementation (Server-Side Rendering)

```go
package main

import (
    "fmt"
    "html/template"
    "net/http"
    "time"
)

type VueComponent struct {
    Name       string
    Data       map[string]interface{}
    Methods    map[string]func() interface{}
    Computed   map[string]func() interface{}
    Watch      map[string]func(interface{})
    Template   string
    Lifecycle  []string
}

type VueApp struct {
    Components map[string]*VueComponent
    Routes     map[string]string
    GlobalData map[string]interface{}
}

func NewVueApp() *VueApp {
    return &VueApp{
        Components: make(map[string]*VueComponent),
        Routes:     make(map[string]string),
        GlobalData: make(map[string]interface{}),
    }
}

func (va *VueApp) CreateComponent(name, template string, data map[string]interface{}) *VueComponent {
    component := &VueComponent{
        Name:      name,
        Data:      data,
        Methods:   make(map[string]func() interface{}),
        Computed:  make(map[string]func() interface{}),
        Watch:     make(map[string]func(interface{})),
        Template:  template,
        Lifecycle: make([]string, 0),
    }
    
    va.Components[name] = component
    va.addLifecycleEvent(component, "beforeCreate")
    va.addLifecycleEvent(component, "created")
    
    fmt.Printf("Created Vue component: %s\n", name)
    return component
}

func (va *VueApp) addLifecycleEvent(component *VueComponent, event string) {
    component.Lifecycle = append(component.Lifecycle, event)
    fmt.Printf("Component %s: %s\n", component.Name, event)
}

func (va *VueApp) AddMethod(componentName, methodName string, method func() interface{}) {
    if component, exists := va.Components[componentName]; exists {
        component.Methods[methodName] = method
        fmt.Printf("Added method %s to component %s\n", methodName, componentName)
    }
}

func (va *VueApp) AddComputed(componentName, computedName string, computed func() interface{}) {
    if component, exists := va.Components[componentName]; exists {
        component.Computed[computedName] = computed
        fmt.Printf("Added computed property %s to component %s\n", computedName, componentName)
    }
}

func (va *VueApp) AddWatcher(componentName, watchName string, watcher func(interface{})) {
    if component, exists := va.Components[componentName]; exists {
        component.Watch[watchName] = watcher
        fmt.Printf("Added watcher for %s to component %s\n", watchName, componentName)
    }
}

func (va *VueApp) MountComponent(componentName string) {
    if component, exists := va.Components[componentName]; exists {
        va.addLifecycleEvent(component, "beforeMount")
        va.addLifecycleEvent(component, "mounted")
    }
}

func (va *VueApp) UpdateComponent(componentName string, newData map[string]interface{}) {
    if component, exists := va.Components[componentName]; exists {
        va.addLifecycleEvent(component, "beforeUpdate")
        
        // Update data
        for key, value := range newData {
            component.Data[key] = value
        }
        
        va.addLifecycleEvent(component, "updated")
    }
}

func (va *VueApp) UnmountComponent(componentName string) {
    if component, exists := va.Components[componentName]; exists {
        va.addLifecycleEvent(component, "beforeDestroy")
        va.addLifecycleEvent(component, "destroyed")
        delete(va.Components, componentName)
    }
}

func (va *VueApp) RenderComponent(componentName string) string {
    if component, exists := va.Components[componentName]; exists {
        va.addLifecycleEvent(component, "render")
        
        // Process template with data
        html := component.Template
        
        // Replace data bindings
        for key, value := range component.Data {
            placeholder := fmt.Sprintf("{{ %s }}", key)
            html = fmt.Sprintf(html, value)
        }
        
        // Replace computed properties
        for key, computed := range component.Computed {
            placeholder := fmt.Sprintf("{{ %s }}", key)
            value := computed()
            html = fmt.Sprintf(html, value)
        }
        
        return html
    }
    
    return ""
}

func (va *VueApp) AddRoute(path, componentName string) {
    va.Routes[path] = componentName
    fmt.Printf("Added route: %s -> %s\n", path, componentName)
}

func (va *VueApp) HandleRequest(w http.ResponseWriter, r *http.Request) {
    path := r.URL.Path
    
    if componentName, exists := va.Routes[path]; exists {
        html := va.RenderComponent(componentName)
        
        // Serve the rendered component
        w.Header().Set("Content-Type", "text/html")
        fmt.Fprintf(w, `
<!DOCTYPE html>
<html>
<head>
    <title>Vue App</title>
    <style>
        .component { border: 1px solid #ccc; margin: 10px; padding: 10px; }
        .app { background-color: #f0f0f0; }
        .header { background-color: #42b883; color: white; }
        .content { background-color: white; }
    </style>
</head>
<body>
    %s
</body>
</html>`, html)
    } else {
        http.NotFound(w, r)
    }
}

func main() {
    app := NewVueApp()
    
    fmt.Println("Vue.js App Demo:")
    
    // Create component with template
    template := `
<div class="app">
    <div class="header">
        <h1>{{ title }}</h1>
    </div>
    <div class="content">
        <p>{{ message }}</p>
        <p>Count: {{ count }}</p>
        <button @click="increment">Increment</button>
    </div>
</div>`
    
    app.CreateComponent("App", template, map[string]interface{}{
        "title":   "My Vue App",
        "message": "Hello, Vue!",
        "count":   0,
    })
    
    // Add methods
    app.AddMethod("App", "increment", func() interface{} {
        // In a real Vue app, this would update the count
        fmt.Println("Increment button clicked")
        return nil
    })
    
    // Add computed properties
    app.AddComputed("App", "doubleCount", func() interface{} {
        if count, exists := app.Components["App"].Data["count"]; exists {
            return count.(int) * 2
        }
        return 0
    })
    
    // Add watchers
    app.AddWatcher("App", "count", func(newValue interface{}) {
        fmt.Printf("Count changed to: %v\n", newValue)
    })
    
    // Mount component
    app.MountComponent("App")
    
    // Update component
    app.UpdateComponent("App", map[string]interface{}{
        "count": 5,
    })
    
    // Add routes
    app.AddRoute("/", "App")
    app.AddRoute("/home", "App")
    
    // Render component
    html := app.RenderComponent("App")
    fmt.Printf("Rendered HTML:\n%s\n", html)
    
    // Start HTTP server
    http.HandleFunc("/", app.HandleRequest)
    fmt.Println("Server starting on :8080")
    go http.ListenAndServe(":8080", nil)
    
    // Keep the program running
    time.Sleep(1 * time.Second)
}
```

## Angular Fundamentals

### Theory

Angular is a TypeScript-based web application framework that provides a complete solution for building large-scale applications. It includes features like dependency injection, routing, and forms.

### Angular Component Implementation

#### Golang Implementation (Server-Side Rendering)

```go
package main

import (
    "fmt"
    "html/template"
    "net/http"
    "reflect"
    "time"
)

type AngularComponent struct {
    Name        string
    Selector    string
    Template    string
    Data        map[string]interface{}
    Services    map[string]interface{}
    Lifecycle   []string
}

type AngularService struct {
    Name    string
    Methods map[string]func() interface{}
}

type AngularApp struct {
    Components map[string]*AngularComponent
    Services   map[string]*AngularService
    Routes     map[string]string
    Modules    map[string][]string
}

func NewAngularApp() *AngularApp {
    return &AngularApp{
        Components: make(map[string]*AngularComponent),
        Services:   make(map[string]*AngularService),
        Routes:     make(map[string]string),
        Modules:    make(map[string][]string),
    }
}

func (aa *AngularApp) CreateService(name string) *AngularService {
    service := &AngularService{
        Name:    name,
        Methods: make(map[string]func() interface{}),
    }
    
    aa.Services[name] = service
    fmt.Printf("Created Angular service: %s\n", name)
    return service
}

func (aa *AngularApp) AddServiceMethod(serviceName, methodName string, method func() interface{}) {
    if service, exists := aa.Services[serviceName]; exists {
        service.Methods[methodName] = method
        fmt.Printf("Added method %s to service %s\n", methodName, serviceName)
    }
}

func (aa *AngularApp) CreateComponent(name, selector, template string, data map[string]interface{}) *AngularComponent {
    component := &AngularComponent{
        Name:      name,
        Selector:  selector,
        Template:  template,
        Data:      data,
        Services:  make(map[string]interface{}),
        Lifecycle: make([]string, 0),
    }
    
    aa.Components[name] = component
    aa.addLifecycleEvent(component, "ngOnInit")
    
    fmt.Printf("Created Angular component: %s\n", name)
    return component
}

func (aa *AngularApp) addLifecycleEvent(component *AngularComponent, event string) {
    component.Lifecycle = append(component.Lifecycle, event)
    fmt.Printf("Component %s: %s\n", component.Name, event)
}

func (aa *AngularApp) InjectService(componentName, serviceName string) {
    if component, exists := aa.Components[componentName]; exists {
        if service, exists := aa.Services[serviceName]; exists {
            component.Services[serviceName] = service
            fmt.Printf("Injected service %s into component %s\n", serviceName, componentName)
        }
    }
}

func (aa *AngularApp) MountComponent(componentName string) {
    if component, exists := aa.Components[componentName]; exists {
        aa.addLifecycleEvent(component, "ngAfterViewInit")
    }
}

func (aa *AngularApp) UpdateComponent(componentName string, newData map[string]interface{}) {
    if component, exists := aa.Components[componentName]; exists {
        aa.addLifecycleEvent(component, "ngOnChanges")
        
        // Update data
        for key, value := range newData {
            component.Data[key] = value
        }
        
        aa.addLifecycleEvent(component, "ngDoCheck")
    }
}

func (aa *AngularApp) UnmountComponent(componentName string) {
    if component, exists := aa.Components[componentName]; exists {
        aa.addLifecycleEvent(component, "ngOnDestroy")
        delete(aa.Components, componentName)
    }
}

func (aa *AngularApp) RenderComponent(componentName string) string {
    if component, exists := aa.Components[componentName]; exists {
        aa.addLifecycleEvent(component, "ngOnChanges")
        
        // Process template with data
        html := component.Template
        
        // Replace data bindings
        for key, value := range component.Data {
            placeholder := fmt.Sprintf("{{ %s }}", key)
            html = fmt.Sprintf(html, value)
        }
        
        return html
    }
    
    return ""
}

func (aa *AngularApp) CreateModule(name string, components []string) {
    aa.Modules[name] = components
    fmt.Printf("Created Angular module: %s with components: %v\n", name, components)
}

func (aa *AngularApp) AddRoute(path, componentName string) {
    aa.Routes[path] = componentName
    fmt.Printf("Added route: %s -> %s\n", path, componentName)
}

func (aa *AngularApp) HandleRequest(w http.ResponseWriter, r *http.Request) {
    path := r.URL.Path
    
    if componentName, exists := aa.Routes[path]; exists {
        html := aa.RenderComponent(componentName)
        
        // Serve the rendered component
        w.Header().Set("Content-Type", "text/html")
        fmt.Fprintf(w, `
<!DOCTYPE html>
<html>
<head>
    <title>Angular App</title>
    <style>
        .component { border: 1px solid #ccc; margin: 10px; padding: 10px; }
        .app { background-color: #f0f0f0; }
        .header { background-color: #dd0031; color: white; }
        .content { background-color: white; }
    </style>
</head>
<body>
    %s
</body>
</html>`, html)
    } else {
        http.NotFound(w, r)
    }
}

func main() {
    app := NewAngularApp()
    
    fmt.Println("Angular App Demo:")
    
    // Create services
    userService := app.CreateService("UserService")
    app.AddServiceMethod("UserService", "getUsers", func() interface{} {
        return []string{"Alice", "Bob", "Charlie"}
    })
    
    // Create component with template
    template := `
<app-root>
    <div class="app">
        <div class="header">
            <h1>{{ title }}</h1>
        </div>
        <div class="content">
            <p>{{ message }}</p>
            <p>Users: {{ users }}</p>
        </div>
    </div>
</app-root>`
    
    app.CreateComponent("AppComponent", "app-root", template, map[string]interface{}{
        "title":   "My Angular App",
        "message": "Hello, Angular!",
        "users":   "Loading...",
    })
    
    // Inject service
    app.InjectService("AppComponent", "UserService")
    
    // Mount component
    app.MountComponent("AppComponent")
    
    // Update component
    app.UpdateComponent("AppComponent", map[string]interface{}{
        "users": "Alice, Bob, Charlie",
    })
    
    // Create module
    app.CreateModule("AppModule", []string{"AppComponent"})
    
    // Add routes
    app.AddRoute("/", "AppComponent")
    app.AddRoute("/home", "AppComponent")
    
    // Render component
    html := app.RenderComponent("AppComponent")
    fmt.Printf("Rendered HTML:\n%s\n", html)
    
    // Start HTTP server
    http.HandleFunc("/", app.HandleRequest)
    fmt.Println("Server starting on :8080")
    go http.ListenAndServe(":8080", nil)
    
    // Keep the program running
    time.Sleep(1 * time.Second)
}
```

## State Management

### Theory

State management is crucial for managing application state in complex frontend applications. Common patterns include Redux, Vuex, and NgRx for predictable state updates.

### State Management Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Action struct {
    Type    string
    Payload interface{}
    Timestamp time.Time
}

type State struct {
    Data    map[string]interface{}
    History []Action
    mutex   sync.RWMutex
}

type Reducer func(State, Action) State

type Store struct {
    State    State
    Reducers map[string]Reducer
    mutex    sync.RWMutex
}

func NewStore() *Store {
    return &Store{
        State: State{
            Data:    make(map[string]interface{}),
            History: make([]Action, 0),
        },
        Reducers: make(map[string]Reducer),
    }
}

func (s *Store) AddReducer(actionType string, reducer Reducer) {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    s.Reducers[actionType] = reducer
    fmt.Printf("Added reducer for action type: %s\n", actionType)
}

func (s *Store) Dispatch(action Action) {
    s.mutex.Lock()
    defer s.mutex.Unlock()
    
    action.Timestamp = time.Now()
    s.State.History = append(s.State.History, action)
    
    if reducer, exists := s.Reducers[action.Type]; exists {
        s.State = reducer(s.State, action)
        fmt.Printf("Dispatched action: %s\n", action.Type)
    } else {
        fmt.Printf("No reducer found for action type: %s\n", action.Type)
    }
}

func (s *Store) GetState() State {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    
    return s.State
}

func (s *Store) Subscribe(callback func(State)) {
    // In a real implementation, this would set up a subscription
    fmt.Println("Subscribed to state changes")
}

func (s *Store) GetHistory() []Action {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    
    return s.State.History
}

func main() {
    store := NewStore()
    
    fmt.Println("State Management Demo:")
    
    // Add reducers
    store.AddReducer("ADD_ITEM", func(state State, action Action) State {
        newState := State{
            Data:    make(map[string]interface{}),
            History: state.History,
        }
        
        // Copy existing data
        for k, v := range state.Data {
            newState.Data[k] = v
        }
        
        // Add new item
        if items, exists := newState.Data["items"]; exists {
            if itemList, ok := items.([]string); ok {
                newState.Data["items"] = append(itemList, action.Payload.(string))
            }
        } else {
            newState.Data["items"] = []string{action.Payload.(string)}
        }
        
        return newState
    })
    
    store.AddReducer("REMOVE_ITEM", func(state State, action Action) State {
        newState := State{
            Data:    make(map[string]interface{}),
            History: state.History,
        }
        
        // Copy existing data
        for k, v := range state.Data {
            newState.Data[k] = v
        }
        
        // Remove item
        if items, exists := newState.Data["items"]; exists {
            if itemList, ok := items.([]string); ok {
                var newItems []string
                for _, item := range itemList {
                    if item != action.Payload.(string) {
                        newItems = append(newItems, item)
                    }
                }
                newState.Data["items"] = newItems
            }
        }
        
        return newState
    })
    
    // Dispatch actions
    store.Dispatch(Action{
        Type:    "ADD_ITEM",
        Payload: "Apple",
    })
    
    store.Dispatch(Action{
        Type:    "ADD_ITEM",
        Payload: "Banana",
    })
    
    store.Dispatch(Action{
        Type:    "ADD_ITEM",
        Payload: "Orange",
    })
    
    // Get current state
    state := store.GetState()
    fmt.Printf("Current state: %v\n", state.Data)
    
    // Remove an item
    store.Dispatch(Action{
        Type:    "REMOVE_ITEM",
        Payload: "Banana",
    })
    
    // Get updated state
    state = store.GetState()
    fmt.Printf("Updated state: %v\n", state.Data)
    
    // Get history
    history := store.GetHistory()
    fmt.Printf("Action history: %d actions\n", len(history))
    for i, action := range history {
        fmt.Printf("  %d. %s: %v\n", i+1, action.Type, action.Payload)
    }
}
```

## Follow-up Questions

### 1. React Fundamentals
**Q: What are the key differences between React and Vue.js?**
A: React uses JSX and a virtual DOM, while Vue.js uses templates and reactive data binding. React has a larger ecosystem and more flexibility, while Vue.js is easier to learn and has better built-in features.

### 2. State Management
**Q: When would you use Redux over local component state?**
A: Use Redux when you have complex state that needs to be shared across multiple components, when you need time-travel debugging, or when you want predictable state updates. Use local state for simple, component-specific data.

### 3. Performance Optimization
**Q: What are the main performance optimization techniques for frontend frameworks?**
A: Use virtual scrolling for large lists, implement lazy loading for components, optimize bundle size with code splitting, use memoization for expensive calculations, and implement proper caching strategies.

## Complexity Analysis

| Operation | React | Vue.js | Angular |
|-----------|-------|--------|---------|
| Component Creation | O(1) | O(1) | O(1) |
| State Update | O(1) | O(1) | O(1) |
| Rendering | O(n) | O(n) | O(n) |
| Bundle Size | Medium | Small | Large |

## Applications

1. **React**: Single-page applications, mobile apps (React Native), complex UIs
2. **Vue.js**: Progressive web apps, small to medium applications, rapid prototyping
3. **Angular**: Large enterprise applications, complex business logic, TypeScript projects
4. **State Management**: Complex applications, real-time apps, collaborative tools

---

**Next**: [Backend Development](./backend-development.md) | **Previous**: [Web Development](../README.md) | **Up**: [Web Development](../README.md)
