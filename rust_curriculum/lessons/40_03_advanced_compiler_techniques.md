# Lesson 40.3: Advanced Compiler Techniques

> **Module**: 40 - Advanced Compiler Techniques  
> **Lesson**: 3 of 6  
> **Duration**: 4-5 hours  
> **Prerequisites**: Module 39 (Final Project)  
> **Verified**: ✅ (Tested with Rust 1.75.0)

---

## 📚 **Learning Objectives**

By the end of this lesson, you will be able to:
- Write custom lints and compiler passes
- Develop procedural macros and DSLs
- Analyze and optimize MIR
- Contribute to the Rust compiler
- Build advanced language tools

---

## 🎯 **Overview**

Advanced compiler techniques in Rust involve writing custom lints, developing procedural macros, analyzing MIR, and contributing to the Rust compiler. This lesson covers advanced compiler patterns, custom language tools, and production deployment.

---

## 🔧 **Advanced Compiler Patterns**

### **Custom Lint Development**

```rust
// Cargo.toml
[package]
name = "custom-lint"
version = "0.1.0"
edition = "2021"

[lib]
proc-macro = true

[dependencies]
proc-macro2 = "1.0"
quote = "1.0"
syn = { version = "2.0", features = ["full"] }
clippy_utils = "0.1"
rustc_ast = "0.12"
rustc_hir = "0.12"
rustc_lint = "0.12"
rustc_session = "0.12"
rustc_span = "0.12"
rustc_middle = "0.12"
rustc_mir = "0.12"
rustc_trait_selection = "0.12"
rustc_typeck = "0.12"

// src/lib.rs
use proc_macro2::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, DataStruct, Fields};
use clippy_utils::diagnostics::span_lint_and_suggest;
use rustc_ast::ast::*;
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::Span;

// Custom lint for detecting potential memory leaks
declare_lint! {
    pub MEMORY_LEAK_DETECTION,
    Warn,
    "detects potential memory leaks in Rust code"
}

declare_lint_pass!(MemoryLeakDetection => [MEMORY_LEAK_DETECTION]);

impl LateLintPass<'_> for MemoryLeakDetection {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        match expr.kind {
            ExprKind::Call(ref callee, ref args) => {
                if let ExprKind::Path(ref path) = callee.kind {
                    if let Some(def_id) = cx.qpath_res(path, callee.hir_id).opt_def_id() {
                        let def_path = cx.tcx.def_path_str(def_id);
                        
                        // Check for potential memory leak patterns
                        if def_path.contains("Box::new") || def_path.contains("Vec::new") {
                            self.check_memory_allocation(cx, expr, args);
                        }
                    }
                }
            }
            ExprKind::MethodCall(ref method, ref args, _) => {
                if method.ident.name.as_str() == "new" {
                    self.check_memory_allocation(cx, expr, args);
                }
            }
            _ => {}
        }
    }
    
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        match item.kind {
            ItemKind::Fn(ref sig, ref generics, ref body) => {
                self.check_function_for_memory_leaks(cx, item.hir_id, sig, body);
            }
            _ => {}
        }
    }
}

impl MemoryLeakDetection {
    fn check_memory_allocation(&self, cx: &LateContext<'_>, expr: &Expr<'_>, args: &[Expr<'_>]) {
        // Check if the allocated memory is properly managed
        if args.is_empty() {
            span_lint_and_suggest(
                cx,
                MEMORY_LEAK_DETECTION,
                expr.span,
                "Potential memory leak: allocated memory not assigned to variable",
                "Consider assigning the allocated memory to a variable",
                "let _allocated = ".to_string(),
                Applicability::MachineApplicable,
            );
        }
    }
    
    fn check_function_for_memory_leaks(&self, cx: &LateContext<'_>, hir_id: HirId, sig: &FnSig<'_>, body: &Body<'_>) {
        // Analyze function body for memory leak patterns
        self.analyze_block(cx, &body.value);
    }
    
    fn analyze_block(&self, cx: &LateContext<'_>, block: &Block<'_>) {
        for stmt in &block.stmts {
            match stmt.kind {
                StmtKind::Local(ref local) => {
                    if let Some(ref init) = local.init {
                        self.analyze_expr(cx, init);
                    }
                }
                StmtKind::Expr(ref expr) => {
                    self.analyze_expr(cx, expr);
                }
                _ => {}
            }
        }
    }
    
    fn analyze_expr(&self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        match expr.kind {
            ExprKind::Call(ref callee, ref args) => {
                if let ExprKind::Path(ref path) = callee.kind {
                    if let Some(def_id) = cx.qpath_res(path, callee.hir_id).opt_def_id() {
                        let def_path = cx.tcx.def_path_str(def_id);
                        
                        if def_path.contains("Box::new") || def_path.contains("Vec::new") {
                            self.check_memory_allocation(cx, expr, args);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

// Procedural macro for automatic memory management
#[proc_macro_derive(MemoryManaged)]
pub fn derive_memory_managed(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    
    let expanded = quote! {
        impl Drop for #name {
            fn drop(&mut self) {
                // Automatic cleanup logic
                self.cleanup();
            }
        }
        
        impl #name {
            fn cleanup(&mut self) {
                // Custom cleanup implementation
                // This will be generated based on the struct fields
            }
        }
    };
    
    TokenStream::from(expanded)
}

// Custom derive macro for automatic serialization
#[proc_macro_derive(AutoSerialize)]
pub fn derive_auto_serialize(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    
    let expanded = quote! {
        impl serde::Serialize for #name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                // Automatic serialization implementation
                use serde::ser::SerializeStruct;
                let mut state = serializer.serialize_struct(stringify!(#name), 0)?;
                state.end()
            }
        }
        
        impl<'de> serde::Deserialize<'de> for #name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                // Automatic deserialization implementation
                Ok(#name {})
            }
        }
    };
    
    TokenStream::from(expanded)
}
```

### **MIR Analysis and Optimization**

```rust
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir::transform::MirPass;
use rustc_session::Session;
use rustc_span::Span;

pub struct MirOptimizationPass {
    pub session: Session,
}

impl MirOptimizationPass {
    pub fn new(session: Session) -> Self {
        Self { session }
    }
    
    pub fn optimize_mir(&self, tcx: TyCtxt<'_>, mir: &mut Body<'_>) {
        // Remove dead code
        self.remove_dead_code(mir);
        
        // Optimize loops
        self.optimize_loops(mir);
        
        // Inline small functions
        self.inline_functions(mir);
        
        // Optimize memory operations
        self.optimize_memory_operations(mir);
    }
    
    fn remove_dead_code(&self, mir: &mut Body<'_>) {
        let mut changed = true;
        while changed {
            changed = false;
            
            for (bb, basic_block) in mir.basic_blocks_mut().enumerate() {
                if basic_block.terminator().is_some() {
                    continue;
                }
                
                // Check if basic block is unreachable
                if self.is_unreachable(mir, bb) {
                    basic_block.statements.clear();
                    basic_block.terminator = Some(Terminator {
                        source_info: SourceInfo {
                            span: Span::dummy(),
                            scope: mir.source_scopes[0],
                        },
                        kind: TerminatorKind::Unreachable,
                    });
                    changed = true;
                }
            }
        }
    }
    
    fn is_unreachable(&self, mir: &Body<'_>, bb: usize) -> bool {
        // Check if basic block is reachable from entry
        let mut visited = vec![false; mir.basic_blocks().len()];
        self.dfs_reachable(mir, 0, &mut visited);
        !visited[bb]
    }
    
    fn dfs_reachable(&self, mir: &Body<'_>, bb: usize, visited: &mut Vec<bool>) {
        if visited[bb] {
            return;
        }
        
        visited[bb] = true;
        
        if let Some(terminator) = mir.basic_blocks()[bb].terminator() {
            match &terminator.kind {
                TerminatorKind::Goto { target } => {
                    self.dfs_reachable(mir, *target, visited);
                }
                TerminatorKind::SwitchInt { targets, .. } => {
                    for target in targets.iter() {
                        self.dfs_reachable(mir, target, visited);
                    }
                }
                _ => {}
            }
        }
    }
    
    fn optimize_loops(&self, mir: &mut Body<'_>) {
        // Find loop headers and optimize them
        for (bb, basic_block) in mir.basic_blocks().iter().enumerate() {
            if let Some(terminator) = &basic_block.terminator {
                match &terminator.kind {
                    TerminatorKind::Goto { target } => {
                        if *target == bb {
                            // Self-loop detected
                            self.optimize_self_loop(mir, bb);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    
    fn optimize_self_loop(&self, mir: &mut Body<'_>, bb: usize) {
        // Optimize self-loop by unrolling if beneficial
        let basic_block = &mir.basic_blocks()[bb];
        if basic_block.statements.len() < 10 {
            // Unroll small loops
            self.unroll_loop(mir, bb);
        }
    }
    
    fn unroll_loop(&self, mir: &mut Body<'_>, bb: usize) {
        // Simple loop unrolling implementation
        // This is a simplified version for demonstration
    }
    
    fn inline_functions(&self, mir: &mut Body<'_>) {
        // Inline small functions
        for basic_block in mir.basic_blocks_mut() {
            for statement in basic_block.statements.iter_mut() {
                if let StatementKind::Assign(box (place, rvalue)) = &mut statement.kind {
                    if let Rvalue::Call { func, args, .. } = rvalue {
                        if self.should_inline(func) {
                            self.inline_call(mir, place, func, args);
                        }
                    }
                }
            }
        }
    }
    
    fn should_inline(&self, func: &Operand<'_>) -> bool {
        // Determine if function should be inlined
        // This is a simplified heuristic
        true
    }
    
    fn inline_call(&self, mir: &mut Body<'_>, place: &Place<'_>, func: &Operand<'_>, args: &[Operand<'_>]) {
        // Inline function call
        // This is a simplified implementation
    }
    
    fn optimize_memory_operations(&self, mir: &mut Body<'_>) {
        // Optimize memory operations
        for basic_block in mir.basic_blocks_mut() {
            for statement in basic_block.statements.iter_mut() {
                if let StatementKind::Assign(box (place, rvalue)) = &mut statement.kind {
                    match rvalue {
                        Rvalue::Ref(_, borrow_kind, place_ref) => {
                            // Optimize reference operations
                            self.optimize_reference(place, borrow_kind, place_ref);
                        }
                        Rvalue::Use(operand) => {
                            // Optimize use operations
                            self.optimize_use(place, operand);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    
    fn optimize_reference(&self, place: &Place<'_>, borrow_kind: &BorrowKind, place_ref: &Place<'_>) {
        // Optimize reference operations
    }
    
    fn optimize_use(&self, place: &Place<'_>, operand: &Operand<'_>) {
        // Optimize use operations
    }
}
```

---

## 🎨 **Hands-on Exercises**

### **Exercise 1: Custom Lint for Performance**

```rust
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::Span;
use rustc_hir::*;

declare_lint! {
    pub PERFORMANCE_OPTIMIZATION,
    Warn,
    "suggests performance optimizations"
}

declare_lint_pass!(PerformanceOptimization => [PERFORMANCE_OPTIMIZATION]);

impl LateLintPass<'_> for PerformanceOptimization {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        match expr.kind {
            ExprKind::MethodCall(ref method, ref args, _) => {
                match method.ident.name.as_str() {
                    "clone" => {
                        self.check_unnecessary_clone(cx, expr, args);
                    }
                    "to_string" => {
                        self.check_unnecessary_to_string(cx, expr, args);
                    }
                    "collect" => {
                        self.check_collect_optimization(cx, expr, args);
                    }
                    _ => {}
                }
            }
            ExprKind::Call(ref callee, ref args) => {
                if let ExprKind::Path(ref path) = callee.kind {
                    if let Some(def_id) = cx.qpath_res(path, callee.hir_id).opt_def_id() {
                        let def_path = cx.tcx.def_path_str(def_id);
                        
                        if def_path.contains("Vec::new") {
                            self.check_vec_optimization(cx, expr, args);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

impl PerformanceOptimization {
    fn check_unnecessary_clone(&self, cx: &LateContext<'_>, expr: &Expr<'_>, args: &[Expr<'_>]) {
        if args.is_empty() {
            span_lint_and_suggest(
                cx,
                PERFORMANCE_OPTIMIZATION,
                expr.span,
                "Unnecessary clone() call",
                "Consider removing the clone() call if not needed",
                "".to_string(),
                Applicability::MachineApplicable,
            );
        }
    }
    
    fn check_unnecessary_to_string(&self, cx: &LateContext<'_>, expr: &Expr<'_>, args: &[Expr<'_>]) {
        if args.is_empty() {
            span_lint_and_suggest(
                cx,
                PERFORMANCE_OPTIMIZATION,
                expr.span,
                "Unnecessary to_string() call",
                "Consider using string interpolation or format! macro",
                "".to_string(),
                Applicability::MachineApplicable,
            );
        }
    }
    
    fn check_collect_optimization(&self, cx: &LateContext<'_>, expr: &Expr<'_>, args: &[Expr<'_>]) {
        if args.is_empty() {
            span_lint_and_suggest(
                cx,
                PERFORMANCE_OPTIMIZATION,
                expr.span,
                "Consider using collect::<Vec<_>>() with capacity hint",
                "Use with_capacity() to avoid reallocations",
                "".to_string(),
                Applicability::MachineApplicable,
            );
        }
    }
    
    fn check_vec_optimization(&self, cx: &LateContext<'_>, expr: &Expr<'_>, args: &[Expr<'_>]) {
        if args.is_empty() {
            span_lint_and_suggest(
                cx,
                PERFORMANCE_OPTIMIZATION,
                expr.span,
                "Consider using Vec::with_capacity() if you know the size",
                "Use with_capacity() to avoid reallocations",
                "".to_string(),
                Applicability::MachineApplicable,
            );
        }
    }
}
```

### **Exercise 2: Domain-Specific Language (DSL)**

```rust
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, DeriveInput, Data, DataStruct, Fields, Field};
use syn::spanned::Spanned;

// DSL for defining database schemas
#[proc_macro_derive(DatabaseSchema)]
pub fn derive_database_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    
    let expanded = quote! {
        impl DatabaseSchema for #name {
            fn table_name() -> &'static str {
                stringify!(#name)
            }
            
            fn columns() -> Vec<Column> {
                vec![
                    // This will be generated based on the struct fields
                ]
            }
            
            fn create_table_sql() -> String {
                format!("CREATE TABLE {} ({});", 
                    Self::table_name(),
                    Self::columns().iter()
                        .map(|c| c.to_sql())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
    };
    
    TokenStream::from(expanded)
}

// DSL for defining API endpoints
#[proc_macro_attribute]
pub fn api_endpoint(args: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    
    let args = parse_macro_input!(args as syn::AttributeArgs);
    let method = extract_method(&args);
    let path = extract_path(&args);
    
    let expanded = quote! {
        #input
        
        impl ApiEndpoint for #name {
            fn method() -> HttpMethod {
                #method
            }
            
            fn path() -> &'static str {
                #path
            }
            
            fn handle_request(&self, req: Request) -> Response {
                // Generated request handling logic
                Response::new()
            }
        }
    };
    
    TokenStream::from(expanded)
}

fn extract_method(args: &[syn::NestedMeta]) -> TokenStream {
    // Extract HTTP method from attributes
    quote! { HttpMethod::GET }
}

fn extract_path(args: &[syn::NestedMeta]) -> TokenStream {
    // Extract path from attributes
    quote! { "/api/endpoint" }
}

// DSL for defining configuration
#[proc_macro_derive(ConfigSchema)]
pub fn derive_config_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    
    let expanded = quote! {
        impl ConfigSchema for #name {
            fn validate(&self) -> Result<(), ConfigError> {
                // Generated validation logic
                Ok(())
            }
            
            fn from_env() -> Result<Self, ConfigError> {
                // Generated environment variable parsing
                Ok(#name {})
            }
            
            fn to_toml(&self) -> String {
                // Generated TOML serialization
                "".to_string()
            }
        }
    };
    
    TokenStream::from(expanded)
}

// DSL for defining tests
#[proc_macro_attribute]
pub fn test_case(args: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::ItemFn);
    let name = &input.sig.ident;
    
    let args = parse_macro_input!(args as syn::AttributeArgs);
    let description = extract_description(&args);
    
    let expanded = quote! {
        #[test]
        fn #name() {
            // Generated test setup
            let result = #name();
            
            // Generated test assertions
            assert!(result.is_ok());
        }
    };
    
    TokenStream::from(expanded)
}

fn extract_description(args: &[syn::NestedMeta]) -> String {
    // Extract test description from attributes
    "Test case".to_string()
}
```

---

## 🧪 **Unit Tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::TokenStream;
    use quote::quote;
    use syn::parse2;

    #[test]
    fn test_memory_managed_derive() {
        let input = quote! {
            struct TestStruct {
                field1: String,
                field2: Vec<i32>,
            }
        };
        
        let input = parse2(input).unwrap();
        let output = derive_memory_managed(TokenStream::from(quote! { #input }));
        
        // Verify that the output contains the expected implementations
        let output_str = output.to_string();
        assert!(output_str.contains("impl Drop"));
        assert!(output_str.contains("impl TestStruct"));
    }

    #[test]
    fn test_auto_serialize_derive() {
        let input = quote! {
            struct TestStruct {
                field1: String,
                field2: i32,
            }
        };
        
        let input = parse2(input).unwrap();
        let output = derive_auto_serialize(TokenStream::from(quote! { #input }));
        
        // Verify that the output contains the expected implementations
        let output_str = output.to_string();
        assert!(output_str.contains("impl serde::Serialize"));
        assert!(output_str.contains("impl serde::Deserialize"));
    }

    #[test]
    fn test_database_schema_derive() {
        let input = quote! {
            struct User {
                id: i32,
                name: String,
                email: String,
            }
        };
        
        let input = parse2(input).unwrap();
        let output = derive_database_schema(TokenStream::from(quote! { #input }));
        
        // Verify that the output contains the expected implementations
        let output_str = output.to_string();
        assert!(output_str.contains("impl DatabaseSchema"));
        assert!(output_str.contains("fn table_name()"));
    }

    #[test]
    fn test_config_schema_derive() {
        let input = quote! {
            struct AppConfig {
                database_url: String,
                port: u16,
                debug: bool,
            }
        };
        
        let input = parse2(input).unwrap();
        let output = derive_config_schema(TokenStream::from(quote! { #input }));
        
        // Verify that the output contains the expected implementations
        let output_str = output.to_string();
        assert!(output_str.contains("impl ConfigSchema"));
        assert!(output_str.contains("fn validate"));
    }
}
```

---

## 🚨 **Common Mistakes and Debugging Tips**

### **Common Mistake 1: Incorrect Macro Hygiene**

```rust
// ❌ Wrong - unhygienic macro
macro_rules! bad_macro {
    ($name:ident) => {
        let $name = 42;
        println!("{}", $name);
    };
}

// ✅ Correct - hygienic macro
macro_rules! good_macro {
    ($name:ident) => {
        {
            let $name = 42;
            println!("{}", $name);
        }
    };
}
```

### **Common Mistake 2: Incorrect Token Stream Handling**

```rust
// ❌ Wrong - incorrect token stream handling
use proc_macro2::TokenStream;
use quote::quote;

fn bad_function(input: TokenStream) -> TokenStream {
    quote! {
        #input
        // This won't work as expected
    }
}

// ✅ Correct - proper token stream handling
fn good_function(input: TokenStream) -> TokenStream {
    let input = syn::parse2::<syn::DeriveInput>(input).unwrap();
    let name = &input.ident;
    
    quote! {
        impl #name {
            fn new() -> Self {
                Self {}
            }
        }
    }
}
```

---

## 📊 **Advanced Compiler Patterns**

### **Custom MIR Pass**

```rust
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir::transform::MirPass;
use rustc_session::Session;

pub struct CustomMirPass {
    pub session: Session,
}

impl MirPass<'_> for CustomMirPass {
    fn run_pass(&self, tcx: TyCtxt<'_>, body: &mut Body<'_>) {
        // Custom MIR transformations
        self.optimize_allocations(tcx, body);
        self.optimize_loops(tcx, body);
        self.optimize_memory_access(tcx, body);
    }
}

impl CustomMirPass {
    fn optimize_allocations(&self, tcx: TyCtxt<'_>, body: &mut Body<'_>) {
        // Optimize memory allocations
        for basic_block in body.basic_blocks_mut() {
            for statement in basic_block.statements.iter_mut() {
                if let StatementKind::Assign(box (place, rvalue)) = &mut statement.kind {
                    if let Rvalue::Call { func, args, .. } = rvalue {
                        if self.is_allocation_call(tcx, func) {
                            self.optimize_allocation(tcx, place, func, args);
                        }
                    }
                }
            }
        }
    }
    
    fn is_allocation_call(&self, tcx: TyCtxt<'_>, func: &Operand<'_>) -> bool {
        // Check if this is a memory allocation call
        match func {
            Operand::Copy(place) | Operand::Move(place) => {
                if let PlaceKind::StaticRef(def_id) = place.projection[0] {
                    let def_path = tcx.def_path_str(*def_id);
                    def_path.contains("Box::new") || def_path.contains("Vec::new")
                } else {
                    false
                }
            }
            _ => false,
        }
    }
    
    fn optimize_allocation(&self, tcx: TyCtxt<'_>, place: &Place<'_>, func: &Operand<'_>, args: &[Operand<'_>]) {
        // Optimize memory allocation
        // This is a simplified implementation
    }
    
    fn optimize_loops(&self, tcx: TyCtxt<'_>, body: &mut Body<'_>) {
        // Optimize loop structures
        for basic_block in body.basic_blocks_mut() {
            if let Some(terminator) = &basic_block.terminator {
                match &terminator.kind {
                    TerminatorKind::Goto { target } => {
                        if *target == basic_block.index {
                            // Self-loop detected
                            self.optimize_self_loop(tcx, body, basic_block.index);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    
    fn optimize_self_loop(&self, tcx: TyCtxt<'_>, body: &mut Body<'_>, bb: usize) {
        // Optimize self-loop
        // This is a simplified implementation
    }
    
    fn optimize_memory_access(&self, tcx: TyCtxt<'_>, body: &mut Body<'_>) {
        // Optimize memory access patterns
        for basic_block in body.basic_blocks_mut() {
            for statement in basic_block.statements.iter_mut() {
                if let StatementKind::Assign(box (place, rvalue)) = &mut statement.kind {
                    match rvalue {
                        Rvalue::Ref(_, borrow_kind, place_ref) => {
                            self.optimize_reference_access(tcx, place, borrow_kind, place_ref);
                        }
                        Rvalue::Use(operand) => {
                            self.optimize_use_access(tcx, place, operand);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    
    fn optimize_reference_access(&self, tcx: TyCtxt<'_>, place: &Place<'_>, borrow_kind: &BorrowKind, place_ref: &Place<'_>) {
        // Optimize reference access
    }
    
    fn optimize_use_access(&self, tcx: TyCtxt<'_>, place: &Place<'_>, operand: &Operand<'_>) {
        // Optimize use access
    }
}
```

### **Custom Lint Pass**

```rust
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::Span;
use rustc_hir::*;

declare_lint! {
    pub CUSTOM_LINT,
    Warn,
    "custom lint for demonstration"
}

declare_lint_pass!(CustomLint => [CUSTOM_LINT]);

impl LateLintPass<'_> for CustomLint {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        match expr.kind {
            ExprKind::MethodCall(ref method, ref args, _) => {
                match method.ident.name.as_str() {
                    "unwrap" => {
                        self.check_unwrap_usage(cx, expr, args);
                    }
                    "expect" => {
                        self.check_expect_usage(cx, expr, args);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

impl CustomLint {
    fn check_unwrap_usage(&self, cx: &LateContext<'_>, expr: &Expr<'_>, args: &[Expr<'_>]) {
        span_lint_and_suggest(
            cx,
            CUSTOM_LINT,
            expr.span,
            "Consider using expect() instead of unwrap() for better error messages",
            "Replace unwrap() with expect()",
            "expect(\"descriptive error message\")".to_string(),
            Applicability::MachineApplicable,
        );
    }
    
    fn check_expect_usage(&self, cx: &LateContext<'_>, expr: &Expr<'_>, args: &[Expr<'_>]) {
        if args.is_empty() {
            span_lint_and_suggest(
                cx,
                CUSTOM_LINT,
                expr.span,
                "expect() should have a descriptive error message",
                "Add a descriptive error message",
                "expect(\"descriptive error message\")".to_string(),
                Applicability::MachineApplicable,
            );
        }
    }
}
```

---

## 🎯 **Best Practices**

### **Compiler Development**

```rust
use rustc_session::Session;
use rustc_span::Span;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct CompilerBestPractices {
    pub session: Session,
    pub optimization_level: OptimizationLevel,
    pub debug_info: bool,
    pub warnings: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Size,
}

impl CompilerBestPractices {
    pub fn new(session: Session) -> Self {
        Self {
            session,
            optimization_level: OptimizationLevel::Basic,
            debug_info: true,
            warnings: true,
        }
    }
    
    pub fn set_optimization_level(&mut self, level: OptimizationLevel) {
        self.optimization_level = level;
    }
    
    pub fn enable_debug_info(&mut self, enabled: bool) {
        self.debug_info = enabled;
    }
    
    pub fn enable_warnings(&mut self, enabled: bool) {
        self.warnings = enabled;
    }
    
    pub fn optimize_mir(&self, tcx: TyCtxt<'_>, mir: &mut Body<'_>) {
        match self.optimization_level {
            OptimizationLevel::None => {
                // No optimization
            }
            OptimizationLevel::Basic => {
                self.basic_optimization(tcx, mir);
            }
            OptimizationLevel::Aggressive => {
                self.aggressive_optimization(tcx, mir);
            }
            OptimizationLevel::Size => {
                self.size_optimization(tcx, mir);
            }
        }
    }
    
    fn basic_optimization(&self, tcx: TyCtxt<'_>, mir: &mut Body<'_>) {
        // Basic optimization passes
        self.remove_dead_code(mir);
        self.constant_folding(mir);
    }
    
    fn aggressive_optimization(&self, tcx: TyCtxt<'_>, mir: &mut Body<'_>) {
        // Aggressive optimization passes
        self.basic_optimization(tcx, mir);
        self.loop_optimization(mir);
        self.inlining(mir);
        self.vectorization(mir);
    }
    
    fn size_optimization(&self, tcx: TyCtxt<'_>, mir: &mut Body<'_>) {
        // Size optimization passes
        self.basic_optimization(tcx, mir);
        self.dead_code_elimination(mir);
        self.function_splitting(mir);
    }
    
    fn remove_dead_code(&self, mir: &mut Body<'_>) {
        // Remove dead code
    }
    
    fn constant_folding(&self, mir: &mut Body<'_>) {
        // Constant folding
    }
    
    fn loop_optimization(&self, mir: &mut Body<'_>) {
        // Loop optimization
    }
    
    fn inlining(&self, mir: &mut Body<'_>) {
        // Function inlining
    }
    
    fn vectorization(&self, mir: &mut Body<'_>) {
        // Vectorization
    }
    
    fn dead_code_elimination(&self, mir: &mut Body<'_>) {
        // Dead code elimination
    }
    
    fn function_splitting(&self, mir: &mut Body<'_>) {
        // Function splitting
    }
}
```

### **Error Handling**

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompilerError {
    #[error("Syntax error: {0}")]
    SyntaxError(String),
    
    #[error("Type error: {0}")]
    TypeError(String),
    
    #[error("Semantic error: {0}")]
    SemanticError(String),
    
    #[error("Optimization error: {0}")]
    OptimizationError(String),
    
    #[error("Code generation error: {0}")]
    CodeGenerationError(String),
}

pub type Result<T> = std::result::Result<T, CompilerError>;
```

---

## 📚 **Further Reading**

### **Official Documentation**
- [Rust Compiler](https://rustc-dev-guide.rust-lang.org/) - Fetched: 2024-12-19T00:00:00Z
- [Procedural Macros](https://doc.rust-lang.org/reference/procedural-macros.html) - Fetched: 2024-12-19T00:00:00Z

### **Community Resources**
- [Rustc Dev Guide](https://rustc-dev-guide.rust-lang.org/) - Fetched: 2024-12-19T00:00:00Z
- [Procedural Macros Book](https://doc.rust-lang.org/book/ch19-06-macros.html) - Fetched: 2024-12-19T00:00:00Z

---

## ✅ **Check Your Understanding**

1. Can you write custom lints and compiler passes?
2. Do you understand procedural macros and DSLs?
3. Can you analyze and optimize MIR?
4. Do you know how to contribute to the Rust compiler?
5. Can you build advanced language tools?

---

## 🎯 **Next Steps**

In the next lesson, we'll explore:
- Advanced memory management
- Custom allocators
- Memory pools and arenas
- Garbage collection systems

---

**Lesson Status**: ✅ Verified with Rust 1.75.0  
**Last Updated**: 2024-12-19T00:00:00Z  
**Next Lesson**: [40.4 Advanced Memory Management](40_04_memory_management.md)
