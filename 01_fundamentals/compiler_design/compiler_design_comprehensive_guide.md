# 🔧 Compiler Design & Language Implementation

## Table of Contents
1. [Lexical Analysis](#lexical-analysis)
2. [Syntax Analysis](#syntax-analysis)
3. [Semantic Analysis](#semantic-analysis)
4. [Intermediate Code Generation](#intermediate-code-generation)
5. [Code Optimization](#code-optimization)
6. [Code Generation](#code-generation)
7. [Runtime Systems](#runtime-systems)
8. [Go Implementation Examples](#go-implementation-examples)
9. [Interview Questions](#interview-questions)

## Lexical Analysis

### Token Definition and Recognition

```go
package main

import (
    "fmt"
    "regexp"
    "strings"
)

type TokenType int

const (
    // Keywords
    IF TokenType = iota
    ELSE
    WHILE
    FOR
    FUNCTION
    RETURN
    VAR
    LET
    CONST
    
    // Literals
    IDENTIFIER
    NUMBER
    STRING
    BOOLEAN
    
    // Operators
    PLUS
    MINUS
    MULTIPLY
    DIVIDE
    ASSIGN
    EQUALS
    NOT_EQUALS
    LESS_THAN
    GREATER_THAN
    LESS_EQUAL
    GREATER_EQUAL
    
    // Delimiters
    LEFT_PAREN
    RIGHT_PAREN
    LEFT_BRACE
    RIGHT_BRACE
    LEFT_BRACKET
    RIGHT_BRACKET
    SEMICOLON
    COMMA
    DOT
    
    // Special
    EOF
    ILLEGAL
)

type Token struct {
    Type    TokenType
    Literal string
    Line    int
    Column  int
}

type Lexer struct {
    input        string
    position     int
    readPosition int
    ch           byte
    line         int
    column       int
}

func NewLexer(input string) *Lexer {
    l := &Lexer{
        input:  input,
        line:   1,
        column: 1,
    }
    l.readChar()
    return l
}

func (l *Lexer) readChar() {
    if l.readPosition >= len(l.input) {
        l.ch = 0
    } else {
        l.ch = l.input[l.readPosition]
    }
    l.position = l.readPosition
    l.readPosition++
    
    if l.ch == '\n' {
        l.line++
        l.column = 1
    } else {
        l.column++
    }
}

func (l *Lexer) peekChar() byte {
    if l.readPosition >= len(l.input) {
        return 0
    }
    return l.input[l.readPosition]
}

func (l *Lexer) NextToken() Token {
    var tok Token
    
    l.skipWhitespace()
    
    tok.Line = l.line
    tok.Column = l.column
    
    switch l.ch {
    case '=':
        if l.peekChar() == '=' {
            l.readChar()
            tok = Token{Type: EQUALS, Literal: "=="}
        } else {
            tok = Token{Type: ASSIGN, Literal: "="}
        }
    case '+':
        tok = Token{Type: PLUS, Literal: "+"}
    case '-':
        tok = Token{Type: MINUS, Literal: "-"}
    case '*':
        tok = Token{Type: MULTIPLY, Literal: "*"}
    case '/':
        tok = Token{Type: DIVIDE, Literal: "/"}
    case '!':
        if l.peekChar() == '=' {
            l.readChar()
            tok = Token{Type: NOT_EQUALS, Literal: "!="}
        } else {
            tok = Token{Type: ILLEGAL, Literal: "!"}
        }
    case '<':
        if l.peekChar() == '=' {
            l.readChar()
            tok = Token{Type: LESS_EQUAL, Literal: "<="}
        } else {
            tok = Token{Type: LESS_THAN, Literal: "<"}
        }
    case '>':
        if l.peekChar() == '=' {
            l.readChar()
            tok = Token{Type: GREATER_EQUAL, Literal: ">="}
        } else {
            tok = Token{Type: GREATER_THAN, Literal: ">"}
        }
    case '(':
        tok = Token{Type: LEFT_PAREN, Literal: "("}
    case ')':
        tok = Token{Type: RIGHT_PAREN, Literal: ")"}
    case '{':
        tok = Token{Type: LEFT_BRACE, Literal: "{"}
    case '}':
        tok = Token{Type: RIGHT_BRACE, Literal: "}"}
    case '[':
        tok = Token{Type: LEFT_BRACKET, Literal: "["}
    case ']':
        tok = Token{Type: RIGHT_BRACKET, Literal: "]"}
    case ';':
        tok = Token{Type: SEMICOLON, Literal: ";"}
    case ',':
        tok = Token{Type: COMMA, Literal: ","}
    case '.':
        tok = Token{Type: DOT, Literal: "."}
    case 0:
        tok = Token{Type: EOF, Literal: ""}
    default:
        if isLetter(l.ch) {
            tok.Literal = l.readIdentifier()
            tok.Type = lookupIdent(tok.Literal)
            return tok
        } else if isDigit(l.ch) {
            tok.Type = NUMBER
            tok.Literal = l.readNumber()
            return tok
        } else {
            tok = Token{Type: ILLEGAL, Literal: string(l.ch)}
        }
    }
    
    l.readChar()
    return tok
}

func (l *Lexer) readIdentifier() string {
    position := l.position
    for isLetter(l.ch) || isDigit(l.ch) {
        l.readChar()
    }
    return l.input[position:l.position]
}

func (l *Lexer) readNumber() string {
    position := l.position
    for isDigit(l.ch) {
        l.readChar()
    }
    return l.input[position:l.position]
}

func (l *Lexer) skipWhitespace() {
    for l.ch == ' ' || l.ch == '\t' || l.ch == '\n' || l.ch == '\r' {
        l.readChar()
    }
}

func isLetter(ch byte) bool {
    return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch == '_'
}

func isDigit(ch byte) bool {
    return '0' <= ch && ch <= '9'
}

var keywords = map[string]TokenType{
    "if":       IF,
    "else":     ELSE,
    "while":    WHILE,
    "for":      FOR,
    "function": FUNCTION,
    "return":   RETURN,
    "var":      VAR,
    "let":      LET,
    "const":    CONST,
    "true":     BOOLEAN,
    "false":    BOOLEAN,
}

func lookupIdent(ident string) TokenType {
    if tok, ok := keywords[ident]; ok {
        return tok
    }
    return IDENTIFIER
}

func main() {
    input := `
        let x = 5;
        let y = 10;
        let z = x + y;
        if (z > 10) {
            return true;
        } else {
            return false;
        }
    `
    
    l := NewLexer(input)
    
    for {
        tok := l.NextToken()
        fmt.Printf("Type: %v, Literal: %q, Line: %d, Column: %d\n", 
                   tok.Type, tok.Literal, tok.Line, tok.Column)
        
        if tok.Type == EOF {
            break
        }
    }
}
```

## Syntax Analysis

### Recursive Descent Parser

```go
package main

import (
    "fmt"
    "strconv"
)

type ASTNode interface {
    String() string
}

type Program struct {
    Statements []Statement
}

type Statement interface {
    ASTNode
    statementNode()
}

type Expression interface {
    ASTNode
    expressionNode()
}

type LetStatement struct {
    Name  *Identifier
    Value Expression
}

func (ls *LetStatement) statementNode() {}
func (ls *LetStatement) String() string {
    return fmt.Sprintf("let %s = %s;", ls.Name.Value, ls.Value.String())
}

type ReturnStatement struct {
    ReturnValue Expression
}

func (rs *ReturnStatement) statementNode() {}
func (rs *ReturnStatement) String() string {
    return fmt.Sprintf("return %s;", rs.ReturnValue.String())
}

type ExpressionStatement struct {
    Expression Expression
}

func (es *ExpressionStatement) statementNode() {}
func (es *ExpressionStatement) String() string {
    return es.Expression.String()
}

type Identifier struct {
    Value string
}

func (i *Identifier) expressionNode() {}
func (i *Identifier) String() string {
    return i.Value
}

type IntegerLiteral struct {
    Value int64
}

func (il *IntegerLiteral) expressionNode() {}
func (il *IntegerLiteral) String() string {
    return strconv.FormatInt(il.Value, 10)
}

type BooleanLiteral struct {
    Value bool
}

func (bl *BooleanLiteral) expressionNode() {}
func (bl *BooleanLiteral) String() string {
    return strconv.FormatBool(bl.Value)
}

type PrefixExpression struct {
    Operator string
    Right    Expression
}

func (pe *PrefixExpression) expressionNode() {}
func (pe *PrefixExpression) String() string {
    return fmt.Sprintf("(%s%s)", pe.Operator, pe.Right.String())
}

type InfixExpression struct {
    Left     Expression
    Operator string
    Right    Expression
}

func (ie *InfixExpression) expressionNode() {}
func (ie *InfixExpression) String() string {
    return fmt.Sprintf("(%s %s %s)", ie.Left.String(), ie.Operator, ie.Right.String())
}

type IfExpression struct {
    Condition   Expression
    Consequence *BlockStatement
    Alternative *BlockStatement
}

func (ie *IfExpression) expressionNode() {}
func (ie *IfExpression) String() string {
    result := fmt.Sprintf("if %s %s", ie.Condition.String(), ie.Consequence.String())
    if ie.Alternative != nil {
        result += fmt.Sprintf(" else %s", ie.Alternative.String())
    }
    return result
}

type BlockStatement struct {
    Statements []Statement
}

func (bs *BlockStatement) statementNode() {}
func (bs *BlockStatement) String() string {
    var result string
    for _, stmt := range bs.Statements {
        result += stmt.String()
    }
    return result
}

type Parser struct {
    lexer        *Lexer
    currentToken Token
    peekToken    Token
    errors       []string
}

func NewParser(lexer *Lexer) *Parser {
    p := &Parser{
        lexer:  lexer,
        errors: []string{},
    }
    
    p.nextToken()
    p.nextToken()
    
    return p
}

func (p *Parser) nextToken() {
    p.currentToken = p.peekToken
    p.peekToken = p.lexer.NextToken()
}

func (p *Parser) ParseProgram() *Program {
    program := &Program{}
    program.Statements = []Statement{}
    
    for p.currentToken.Type != EOF {
        stmt := p.parseStatement()
        if stmt != nil {
            program.Statements = append(program.Statements, stmt)
        }
        p.nextToken()
    }
    
    return program
}

func (p *Parser) parseStatement() Statement {
    switch p.currentToken.Type {
    case LET:
        return p.parseLetStatement()
    case RETURN:
        return p.parseReturnStatement()
    default:
        return p.parseExpressionStatement()
    }
}

func (p *Parser) parseLetStatement() *LetStatement {
    stmt := &LetStatement{}
    
    if !p.expectPeek(IDENTIFIER) {
        return nil
    }
    
    stmt.Name = &Identifier{Value: p.currentToken.Literal}
    
    if !p.expectPeek(ASSIGN) {
        return nil
    }
    
    p.nextToken()
    stmt.Value = p.parseExpression(LOWEST)
    
    if p.peekTokenIs(SEMICOLON) {
        p.nextToken()
    }
    
    return stmt
}

func (p *Parser) parseReturnStatement() *ReturnStatement {
    stmt := &ReturnStatement{}
    
    p.nextToken()
    stmt.ReturnValue = p.parseExpression(LOWEST)
    
    if p.peekTokenIs(SEMICOLON) {
        p.nextToken()
    }
    
    return stmt
}

func (p *Parser) parseExpressionStatement() *ExpressionStatement {
    stmt := &ExpressionStatement{Expression: p.parseExpression(LOWEST)}
    
    if p.peekTokenIs(SEMICOLON) {
        p.nextToken()
    }
    
    return stmt
}

type Precedence int

const (
    LOWEST Precedence = iota
    EQUALS
    LESSGREATER
    SUM
    PRODUCT
    PREFIX
    CALL
)

var precedences = map[TokenType]Precedence{
    EQUALS:       EQUALS,
    NOT_EQUALS:   EQUALS,
    LESS_THAN:    LESSGREATER,
    GREATER_THAN: LESSGREATER,
    LESS_EQUAL:   LESSGREATER,
    GREATER_EQUAL: LESSGREATER,
    PLUS:         SUM,
    MINUS:        SUM,
    MULTIPLY:     PRODUCT,
    DIVIDE:       PRODUCT,
}

func (p *Parser) parseExpression(precedence Precedence) Expression {
    prefix := p.prefixParseFns[p.currentToken.Type]
    if prefix == nil {
        p.noPrefixParseFnError(p.currentToken.Type)
        return nil
    }
    leftExp := prefix()
    
    for !p.peekTokenIs(SEMICOLON) && precedence < p.peekPrecedence() {
        infix := p.infixParseFns[p.peekToken.Type]
        if infix == nil {
            return leftExp
        }
        
        p.nextToken()
        leftExp = infix(leftExp)
    }
    
    return leftExp
}

type prefixParseFn func() Expression
type infixParseFn func(Expression) Expression

func (p *Parser) parseIdentifier() Expression {
    return &Identifier{Value: p.currentToken.Literal}
}

func (p *Parser) parseIntegerLiteral() Expression {
    lit := &IntegerLiteral{}
    
    value, err := strconv.ParseInt(p.currentToken.Literal, 0, 64)
    if err != nil {
        msg := fmt.Sprintf("could not parse %q as integer", p.currentToken.Literal)
        p.errors = append(p.errors, msg)
        return nil
    }
    
    lit.Value = value
    return lit
}

func (p *Parser) parseBoolean() Expression {
    return &BooleanLiteral{Value: p.currentTokenIs(BOOLEAN) && p.currentToken.Literal == "true"}
}

func (p *Parser) parsePrefixExpression() Expression {
    expression := &PrefixExpression{
        Operator: p.currentToken.Literal,
    }
    
    p.nextToken()
    expression.Right = p.parseExpression(PREFIX)
    
    return expression
}

func (p *Parser) parseInfixExpression(left Expression) Expression {
    expression := &InfixExpression{
        Left:     left,
        Operator: p.currentToken.Literal,
    }
    
    precedence := p.currentPrecedence()
    p.nextToken()
    expression.Right = p.parseExpression(precedence)
    
    return expression
}

func (p *Parser) parseGroupedExpression() Expression {
    p.nextToken()
    
    exp := p.parseExpression(LOWEST)
    
    if !p.expectPeek(RIGHT_PAREN) {
        return nil
    }
    
    return exp
}

func (p *Parser) parseIfExpression() Expression {
    expression := &IfExpression{}
    
    if !p.expectPeek(LEFT_PAREN) {
        return nil
    }
    
    p.nextToken()
    expression.Condition = p.parseExpression(LOWEST)
    
    if !p.expectPeek(RIGHT_PAREN) {
        return nil
    }
    
    if !p.expectPeek(LEFT_BRACE) {
        return nil
    }
    
    expression.Consequence = p.parseBlockStatement()
    
    if p.peekTokenIs(ELSE) {
        p.nextToken()
        
        if !p.expectPeek(LEFT_BRACE) {
            return nil
        }
        
        expression.Alternative = p.parseBlockStatement()
    }
    
    return expression
}

func (p *Parser) parseBlockStatement() *BlockStatement {
    block := &BlockStatement{}
    block.Statements = []Statement{}
    
    p.nextToken()
    
    for !p.currentTokenIs(RIGHT_BRACE) && !p.currentTokenIs(EOF) {
        stmt := p.parseStatement()
        if stmt != nil {
            block.Statements = append(block.Statements, stmt)
        }
        p.nextToken()
    }
    
    return block
}

func (p *Parser) currentTokenIs(t TokenType) bool {
    return p.currentToken.Type == t
}

func (p *Parser) peekTokenIs(t TokenType) bool {
    return p.peekToken.Type == t
}

func (p *Parser) expectPeek(t TokenType) bool {
    if p.peekTokenIs(t) {
        p.nextToken()
        return true
    } else {
        p.peekError(t)
        return false
    }
}

func (p *Parser) peekPrecedence() Precedence {
    if p, ok := precedences[p.peekToken.Type]; ok {
        return p
    }
    return LOWEST
}

func (p *Parser) currentPrecedence() Precedence {
    if p, ok := precedences[p.currentToken.Type]; ok {
        return p
    }
    return LOWEST
}

func (p *Parser) peekError(t TokenType) {
    msg := fmt.Sprintf("expected next token to be %s, got %s instead",
        t, p.peekToken.Type)
    p.errors = append(p.errors, msg)
}

func (p *Parser) noPrefixParseFnError(t TokenType) {
    msg := fmt.Sprintf("no prefix parse function for %s found", t)
    p.errors = append(p.errors, msg)
}

func main() {
    input := `
        let x = 5;
        let y = 10;
        let z = x + y;
        if (z > 10) {
            return true;
        } else {
            return false;
        }
    `
    
    l := NewLexer(input)
    p := NewParser(l)
    
    program := p.ParseProgram()
    
    if len(p.errors) > 0 {
        fmt.Printf("Parser errors:\n")
        for _, err := range p.errors {
            fmt.Printf("  %s\n", err)
        }
        return
    }
    
    fmt.Printf("Parsed program:\n")
    for _, stmt := range program.Statements {
        fmt.Printf("  %s\n", stmt.String())
    }
}
```

## Semantic Analysis

### Symbol Table and Type Checking

```go
package main

import (
    "fmt"
    "strings"
)

type ObjectType int

const (
    INTEGER_OBJ ObjectType = iota
    BOOLEAN_OBJ
    NULL_OBJ
    RETURN_VALUE_OBJ
    ERROR_OBJ
    FUNCTION_OBJ
    STRING_OBJ
    BUILTIN_OBJ
    ARRAY_OBJ
    HASH_OBJ
)

type Object interface {
    Type() ObjectType
    Inspect() string
}

type Integer struct {
    Value int64
}

func (i *Integer) Type() ObjectType { return INTEGER_OBJ }
func (i *Integer) Inspect() string  { return fmt.Sprintf("%d", i.Value) }

type Boolean struct {
    Value bool
}

func (b *Boolean) Type() ObjectType { return BOOLEAN_OBJ }
func (b *Boolean) Inspect() string  { return fmt.Sprintf("%t", b.Value) }

type Null struct{}

func (n *Null) Type() ObjectType { return NULL_OBJ }
func (n *Null) Inspect() string  { return "null" }

type ReturnValue struct {
    Value Object
}

func (rv *ReturnValue) Type() ObjectType { return RETURN_VALUE_OBJ }
func (rv *ReturnValue) Inspect() string  { return rv.Value.Inspect() }

type Error struct {
    Message string
}

func (e *Error) Type() ObjectType { return ERROR_OBJ }
func (e *Error) Inspect() string  { return "ERROR: " + e.Message }

type Environment struct {
    store map[string]Object
    outer *Environment
}

func NewEnvironment() *Environment {
    s := make(map[string]Object)
    return &Environment{store: s, outer: nil}
}

func NewEnclosedEnvironment(outer *Environment) *Environment {
    env := NewEnvironment()
    env.outer = outer
    return env
}

func (e *Environment) Get(name string) (Object, bool) {
    obj, ok := e.store[name]
    if !ok && e.outer != nil {
        obj, ok = e.outer.Get(name)
    }
    return obj, ok
}

func (e *Environment) Set(name string, val Object) Object {
    e.store[name] = val
    return val
}

type Evaluator struct {
    env *Environment
}

func NewEvaluator() *Evaluator {
    return &Evaluator{env: NewEnvironment()}
}

func (e *Evaluator) Eval(node ASTNode) Object {
    switch node := node.(type) {
    case *Program:
        return e.evalProgram(node)
    case *ExpressionStatement:
        return e.Eval(node.Expression)
    case *IntegerLiteral:
        return &Integer{Value: node.Value}
    case *BooleanLiteral:
        return &Boolean{Value: node.Value}
    case *PrefixExpression:
        right := e.Eval(node.Right)
        if isError(right) {
            return right
        }
        return e.evalPrefixExpression(node.Operator, right)
    case *InfixExpression:
        left := e.Eval(node.Left)
        if isError(left) {
            return left
        }
        right := e.Eval(node.Right)
        if isError(right) {
            return right
        }
        return e.evalInfixExpression(node.Operator, left, right)
    case *BlockStatement:
        return e.evalBlockStatement(node)
    case *IfExpression:
        return e.evalIfExpression(node)
    case *ReturnStatement:
        val := e.Eval(node.ReturnValue)
        if isError(val) {
            return val
        }
        return &ReturnValue{Value: val}
    case *LetStatement:
        val := e.Eval(node.Value)
        if isError(val) {
            return val
        }
        e.env.Set(node.Name.Value, val)
    case *Identifier:
        return e.evalIdentifier(node)
    default:
        return &Error{Message: fmt.Sprintf("unknown node type: %T", node)}
    }
    return &Null{}
}

func (e *Evaluator) evalProgram(program *Program) Object {
    var result Object
    
    for _, statement := range program.Statements {
        result = e.Eval(statement)
        
        switch result := result.(type) {
        case *ReturnValue:
            return result.Value
        case *Error:
            return result
        }
    }
    
    return result
}

func (e *Evaluator) evalBlockStatement(block *BlockStatement) Object {
    var result Object
    
    for _, statement := range block.Statements {
        result = e.Eval(statement)
        
        if result != nil {
            rt := result.Type()
            if rt == RETURN_VALUE_OBJ || rt == ERROR_OBJ {
                return result
            }
        }
    }
    
    return result
}

func (e *Evaluator) evalPrefixExpression(operator string, right Object) Object {
    switch operator {
    case "!":
        return e.evalBangOperatorExpression(right)
    case "-":
        return e.evalMinusPrefixOperatorExpression(right)
    default:
        return &Error{Message: fmt.Sprintf("unknown operator: %s%s", operator, right.Type())}
    }
}

func (e *Evaluator) evalInfixExpression(operator string, left, right Object) Object {
    switch {
    case left.Type() == INTEGER_OBJ && right.Type() == INTEGER_OBJ:
        return e.evalIntegerInfixExpression(operator, left, right)
    case operator == "==":
        return &Boolean{Value: left == right}
    case operator == "!=":
        return &Boolean{Value: left != right}
    case left.Type() != right.Type():
        return &Error{Message: fmt.Sprintf("type mismatch: %s %s %s", left.Type(), operator, right.Type())}
    default:
        return &Error{Message: fmt.Sprintf("unknown operator: %s %s %s", left.Type(), operator, right.Type())}
    }
}

func (e *Evaluator) evalBangOperatorExpression(right Object) Object {
    switch right {
    case &Boolean{Value: true}:
        return &Boolean{Value: false}
    case &Boolean{Value: false}:
        return &Boolean{Value: true}
    case &Null{}:
        return &Boolean{Value: true}
    default:
        return &Boolean{Value: false}
    }
}

func (e *Evaluator) evalMinusPrefixOperatorExpression(right Object) Object {
    if right.Type() != INTEGER_OBJ {
        return &Error{Message: fmt.Sprintf("unknown operator: -%s", right.Type())}
    }
    
    value := right.(*Integer).Value
    return &Integer{Value: -value}
}

func (e *Evaluator) evalIntegerInfixExpression(operator string, left, right Object) Object {
    leftVal := left.(*Integer).Value
    rightVal := right.(*Integer).Value
    
    switch operator {
    case "+":
        return &Integer{Value: leftVal + rightVal}
    case "-":
        return &Integer{Value: leftVal - rightVal}
    case "*":
        return &Integer{Value: leftVal * rightVal}
    case "/":
        return &Integer{Value: leftVal / rightVal}
    case "<":
        return &Boolean{Value: leftVal < rightVal}
    case ">":
        return &Boolean{Value: leftVal > rightVal}
    case "==":
        return &Boolean{Value: leftVal == rightVal}
    case "!=":
        return &Boolean{Value: leftVal != rightVal}
    default:
        return &Error{Message: fmt.Sprintf("unknown operator: %s %s %s", left.Type(), operator, right.Type())}
    }
}

func (e *Evaluator) evalIfExpression(ie *IfExpression) Object {
    condition := e.Eval(ie.Condition)
    if isError(condition) {
        return condition
    }
    
    if isTruthy(condition) {
        return e.Eval(ie.Consequence)
    } else if ie.Alternative != nil {
        return e.Eval(ie.Alternative)
    } else {
        return &Null{}
    }
}

func (e *Evaluator) evalIdentifier(node *Identifier) Object {
    val, ok := e.env.Get(node.Value)
    if !ok {
        return &Error{Message: fmt.Sprintf("identifier not found: %s", node.Value)}
    }
    return val
}

func isTruthy(obj Object) bool {
    switch obj {
    case &Null{}:
        return false
    case &Boolean{Value: false}:
        return false
    default:
        return true
    }
}

func isError(obj Object) bool {
    if obj != nil {
        return obj.Type() == ERROR_OBJ
    }
    return false
}

func main() {
    input := `
        let x = 5;
        let y = 10;
        let z = x + y;
        if (z > 10) {
            return true;
        } else {
            return false;
        }
    `
    
    l := NewLexer(input)
    p := NewParser(l)
    program := p.ParseProgram()
    
    if len(p.errors) > 0 {
        fmt.Printf("Parser errors:\n")
        for _, err := range p.errors {
            fmt.Printf("  %s\n", err)
        }
        return
    }
    
    evaluator := NewEvaluator()
    result := evaluator.Eval(program)
    
    fmt.Printf("Evaluation result: %s\n", result.Inspect())
}
```

## Interview Questions

### Basic Concepts
1. **What are the phases of a compiler?**
2. **Explain the difference between lexical analysis and syntax analysis.**
3. **What is a symbol table and why is it important?**
4. **How does a recursive descent parser work?**
5. **What is the purpose of intermediate code generation?**

### Advanced Topics
1. **How would you implement a garbage collector?**
2. **Explain different code optimization techniques.**
3. **How do you handle operator precedence in parsing?**
4. **What are the challenges in implementing closures?**
5. **How would you design a JIT compiler?**

### System Design
1. **Design a simple programming language.**
2. **How would you implement a virtual machine?**
3. **Design a transpiler from one language to another.**
4. **How would you implement a debugger?**
5. **Design a language server protocol implementation.**

## Conclusion

Compiler design is a complex field that involves:

- **Lexical Analysis**: Tokenization and pattern recognition
- **Syntax Analysis**: Parsing and AST construction
- **Semantic Analysis**: Type checking and symbol resolution
- **Code Generation**: Target code production
- **Optimization**: Performance improvements
- **Runtime Systems**: Memory management and execution

Key skills for compiler engineers:
- Deep understanding of language theory
- Experience with parsing algorithms
- Knowledge of target architectures
- Understanding of optimization techniques
- Ability to design efficient data structures
- Experience with runtime systems

This guide provides a foundation for understanding compiler design and preparing for compiler engineering interviews.


## Intermediate Code Generation

<!-- AUTO-GENERATED ANCHOR: originally referenced as #intermediate-code-generation -->

Placeholder content. Please replace with proper section.


## Code Optimization

<!-- AUTO-GENERATED ANCHOR: originally referenced as #code-optimization -->

Placeholder content. Please replace with proper section.


## Runtime Systems

<!-- AUTO-GENERATED ANCHOR: originally referenced as #runtime-systems -->

Placeholder content. Please replace with proper section.


## Go Implementation Examples

<!-- AUTO-GENERATED ANCHOR: originally referenced as #go-implementation-examples -->

Placeholder content. Please replace with proper section.
