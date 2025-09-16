# Database Design

## Overview

This module covers database design concepts including normalization, relationships, constraints, indexing strategies, and query optimization. These concepts are essential for designing efficient and scalable database systems.

## Table of Contents

1. [Database Normalization](#database-normalization)
2. [Entity-Relationship Modeling](#entity-relationship-modeling)
3. [Indexing Strategies](#indexing-strategies)
4. [Query Optimization](#query-optimization)
5. [Transaction Management](#transaction-management)
6. [Applications](#applications)
7. [Complexity Analysis](#complexity-analysis)
8. [Follow-up Questions](#follow-up-questions)

## Database Normalization

### Theory

Database normalization is the process of organizing data in a database to reduce redundancy and improve data integrity. The main normal forms are 1NF, 2NF, 3NF, BCNF, and higher normal forms.

### Normalization Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "strings"
)

type Attribute struct {
    Name string
    Type string
}

type FunctionalDependency struct {
    Determinant []string
    Dependent   []string
}

type Table struct {
    Name        string
    Attributes  []Attribute
    PrimaryKey  []string
    Dependencies []FunctionalDependency
}

type DatabaseDesigner struct {
    tables []Table
}

func NewDatabaseDesigner() *DatabaseDesigner {
    return &DatabaseDesigner{
        tables: make([]Table, 0),
    }
}

func (dd *DatabaseDesigner) AddTable(name string, attributes []Attribute, primaryKey []string) *Table {
    table := Table{
        Name:       name,
        Attributes: attributes,
        PrimaryKey: primaryKey,
        Dependencies: make([]FunctionalDependency, 0),
    }
    
    dd.tables = append(dd.tables, table)
    fmt.Printf("Added table: %s\n", name)
    return &dd.tables[len(dd.tables)-1]
}

func (dd *DatabaseDesigner) AddFunctionalDependency(tableName string, determinant, dependent []string) {
    for i, table := range dd.tables {
        if table.Name == tableName {
            fd := FunctionalDependency{
                Determinant: determinant,
                Dependent:   dependent,
            }
            dd.tables[i].Dependencies = append(dd.tables[i].Dependencies, fd)
            fmt.Printf("Added FD to %s: %v -> %v\n", tableName, determinant, dependent)
            break
        }
    }
}

func (dd *DatabaseDesigner) CheckFirstNormalForm(tableName string) bool {
    table := dd.getTable(tableName)
    if table == nil {
        return false
    }
    
    // Check if all attributes are atomic (no multivalued attributes)
    for _, attr := range table.Attributes {
        if strings.Contains(attr.Name, ",") || strings.Contains(attr.Name, ";") {
            fmt.Printf("Table %s violates 1NF: attribute %s is not atomic\n", tableName, attr.Name)
            return false
        }
    }
    
    fmt.Printf("Table %s is in 1NF\n", tableName)
    return true
}

func (dd *DatabaseDesigner) CheckSecondNormalForm(tableName string) bool {
    table := dd.getTable(tableName)
    if table == nil {
        return false
    }
    
    // Check if all non-key attributes are fully functionally dependent on the primary key
    for _, fd := range table.Dependencies {
        // Check if determinant is a proper subset of primary key
        if !dd.isSubset(fd.Determinant, table.PrimaryKey) {
            // Check if any dependent attribute is not in primary key
            for _, dep := range fd.Dependent {
                if !dd.contains(table.PrimaryKey, dep) {
                    fmt.Printf("Table %s violates 2NF: %v -> %v (partial dependency)\n", 
                               tableName, fd.Determinant, fd.Dependent)
                    return false
                }
            }
        }
    }
    
    fmt.Printf("Table %s is in 2NF\n", tableName)
    return true
}

func (dd *DatabaseDesigner) CheckThirdNormalForm(tableName string) bool {
    table := dd.getTable(tableName)
    if table == nil {
        return false
    }
    
    // Check if all non-key attributes are non-transitively dependent on the primary key
    for _, fd := range table.Dependencies {
        // Check if determinant is not a superkey
        if !dd.isSuperkey(fd.Determinant, table) {
            // Check if any dependent attribute is not in primary key
            for _, dep := range fd.Dependent {
                if !dd.contains(table.PrimaryKey, dep) {
                    fmt.Printf("Table %s violates 3NF: %v -> %v (transitive dependency)\n", 
                               tableName, fd.Determinant, fd.Dependent)
                    return false
                }
            }
        }
    }
    
    fmt.Printf("Table %s is in 3NF\n", tableName)
    return true
}

func (dd *DatabaseDesigner) CheckBCNF(tableName string) bool {
    table := dd.getTable(tableName)
    if table == nil {
        return false
    }
    
    // Check if every determinant is a superkey
    for _, fd := range table.Dependencies {
        if !dd.isSuperkey(fd.Determinant, table) {
            fmt.Printf("Table %s violates BCNF: %v -> %v (determinant is not a superkey)\n", 
                       tableName, fd.Determinant, fd.Dependent)
            return false
        }
    }
    
    fmt.Printf("Table %s is in BCNF\n", tableName)
    return true
}

func (dd *DatabaseDesigner) NormalizeTable(tableName string) []Table {
    table := dd.getTable(tableName)
    if table == nil {
        return []Table{}
    }
    
    fmt.Printf("Normalizing table: %s\n", tableName)
    
    // Check current normal form
    dd.CheckFirstNormalForm(tableName)
    dd.CheckSecondNormalForm(tableName)
    dd.CheckThirdNormalForm(tableName)
    dd.CheckBCNF(tableName)
    
    // For simplicity, return the original table
    // In a real implementation, this would perform actual normalization
    return []Table{*table}
}

func (dd *DatabaseDesigner) getTable(name string) *Table {
    for i, table := range dd.tables {
        if table.Name == name {
            return &dd.tables[i]
        }
    }
    return nil
}

func (dd *DatabaseDesigner) isSubset(subset, superset []string) bool {
    for _, item := range subset {
        if !dd.contains(superset, item) {
            return false
        }
    }
    return true
}

func (dd *DatabaseDesigner) contains(slice []string, item string) bool {
    for _, s := range slice {
        if s == item {
            return true
        }
    }
    return false
}

func (dd *DatabaseDesigner) isSuperkey(attributes []string, table *Table) bool {
    // Check if attributes contain the primary key
    return dd.isSubset(table.PrimaryKey, attributes)
}

func (dd *DatabaseDesigner) PrintTable(tableName string) {
    table := dd.getTable(tableName)
    if table == nil {
        fmt.Printf("Table %s not found\n", tableName)
        return
    }
    
    fmt.Printf("Table: %s\n", table.Name)
    fmt.Printf("  Primary Key: %v\n", table.PrimaryKey)
    fmt.Printf("  Attributes:\n")
    for _, attr := range table.Attributes {
        fmt.Printf("    %s (%s)\n", attr.Name, attr.Type)
    }
    fmt.Printf("  Functional Dependencies:\n")
    for _, fd := range table.Dependencies {
        fmt.Printf("    %v -> %v\n", fd.Determinant, fd.Dependent)
    }
}

func main() {
    designer := NewDatabaseDesigner()
    
    fmt.Println("Database Design Demo:")
    
    // Create a sample table
    attributes := []Attribute{
        {Name: "StudentID", Type: "INT"},
        {Name: "StudentName", Type: "VARCHAR(100)"},
        {Name: "CourseID", Type: "INT"},
        {Name: "CourseName", Type: "VARCHAR(100)"},
        {Name: "InstructorID", Type: "INT"},
        {Name: "InstructorName", Type: "VARCHAR(100)"},
        {Name: "Grade", Type: "CHAR(2)"},
    }
    
    primaryKey := []string{"StudentID", "CourseID"}
    
    table := designer.AddTable("StudentCourse", attributes, primaryKey)
    
    // Add functional dependencies
    designer.AddFunctionalDependency("StudentCourse", []string{"StudentID"}, []string{"StudentName"})
    designer.AddFunctionalDependency("StudentCourse", []string{"CourseID"}, []string{"CourseName"})
    designer.AddFunctionalDependency("StudentCourse", []string{"CourseID"}, []string{"InstructorID"})
    designer.AddFunctionalDependency("StudentCourse", []string{"InstructorID"}, []string{"InstructorName"})
    
    // Print table information
    designer.PrintTable("StudentCourse")
    
    // Check normal forms
    fmt.Println("\nNormal Form Analysis:")
    designer.CheckFirstNormalForm("StudentCourse")
    designer.CheckSecondNormalForm("StudentCourse")
    designer.CheckThirdNormalForm("StudentCourse")
    designer.CheckBCNF("StudentCourse")
    
    // Normalize table
    fmt.Println("\nNormalization:")
    normalizedTables := designer.NormalizeTable("StudentCourse")
    fmt.Printf("Normalized into %d tables\n", len(normalizedTables))
}
```

## Entity-Relationship Modeling

### Theory

Entity-Relationship (ER) modeling is a conceptual data modeling technique that describes the structure of a database in terms of entities, relationships, and attributes.

### ER Model Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "strings"
)

type AttributeType int

const (
    Simple AttributeType = iota
    Composite
    SingleValued
    MultiValued
    Derived
)

type Attribute struct {
    Name string
    Type AttributeType
    DataType string
    IsKey bool
}

type Entity struct {
    Name       string
    Attributes []Attribute
    PrimaryKey []string
}

type RelationshipType int

const (
    OneToOne RelationshipType = iota
    OneToMany
    ManyToMany
)

type Relationship struct {
    Name         string
    Type         RelationshipType
    Entities     []string
    Attributes   []Attribute
    Cardinality  map[string]string
}

type ERModel struct {
    Entities     map[string]*Entity
    Relationships map[string]*Relationship
}

func NewERModel() *ERModel {
    return &ERModel{
        Entities:     make(map[string]*Entity),
        Relationships: make(map[string]*Relationship),
    }
}

func (erm *ERModel) AddEntity(name string, attributes []Attribute) *Entity {
    entity := &Entity{
        Name:       name,
        Attributes: attributes,
        PrimaryKey: make([]string, 0),
    }
    
    // Identify primary key attributes
    for _, attr := range attributes {
        if attr.IsKey {
            entity.PrimaryKey = append(entity.PrimaryKey, attr.Name)
        }
    }
    
    erm.Entities[name] = entity
    fmt.Printf("Added entity: %s\n", name)
    return entity
}

func (erm *ERModel) AddRelationship(name string, relType RelationshipType, entities []string, attributes []Attribute) *Relationship {
    relationship := &Relationship{
        Name:        name,
        Type:        relType,
        Entities:    entities,
        Attributes:  attributes,
        Cardinality: make(map[string]string),
    }
    
    erm.Relationships[name] = relationship
    fmt.Printf("Added relationship: %s\n", name)
    return relationship
}

func (erm *ERModel) SetCardinality(relName string, entityName, cardinality string) {
    if rel, exists := erm.Relationships[relName]; exists {
        rel.Cardinality[entityName] = cardinality
        fmt.Printf("Set cardinality for %s in relationship %s: %s\n", entityName, relName, cardinality)
    }
}

func (erm *ERModel) GenerateSQL() []string {
    var sqlStatements []string
    
    // Generate CREATE TABLE statements for entities
    for _, entity := range erm.Entities {
        sql := erm.generateEntitySQL(entity)
        sqlStatements = append(sqlStatements, sql)
    }
    
    // Generate CREATE TABLE statements for relationships
    for _, rel := range erm.Relationships {
        if rel.Type == ManyToMany {
            sql := erm.generateRelationshipSQL(rel)
            sqlStatements = append(sqlStatements, sql)
        }
    }
    
    return sqlStatements
}

func (erm *ERModel) generateEntitySQL(entity *Entity) string {
    var sql strings.Builder
    
    sql.WriteString(fmt.Sprintf("CREATE TABLE %s (\n", entity.Name))
    
    // Add attributes
    for i, attr := range entity.Attributes {
        sql.WriteString(fmt.Sprintf("    %s %s", attr.Name, attr.DataType))
        
        if attr.IsKey {
            sql.WriteString(" PRIMARY KEY")
        }
        
        if i < len(entity.Attributes)-1 {
            sql.WriteString(",")
        }
        sql.WriteString("\n")
    }
    
    sql.WriteString(");")
    return sql.String()
}

func (erm *ERModel) generateRelationshipSQL(rel *Relationship) string {
    var sql strings.Builder
    
    sql.WriteString(fmt.Sprintf("CREATE TABLE %s (\n", rel.Name))
    
    // Add foreign keys for each entity
    for i, entityName := range rel.Entities {
        if entity, exists := erm.Entities[entityName]; exists {
            for j, pk := range entity.PrimaryKey {
                sql.WriteString(fmt.Sprintf("    %s_%s %s", entityName, pk, "INT"))
                if i < len(rel.Entities)-1 || j < len(entity.PrimaryKey)-1 {
                    sql.WriteString(",")
                }
                sql.WriteString("\n")
            }
        }
    }
    
    // Add relationship attributes
    for i, attr := range rel.Attributes {
        sql.WriteString(fmt.Sprintf("    %s %s", attr.Name, attr.DataType))
        if i < len(rel.Attributes)-1 {
            sql.WriteString(",")
        }
        sql.WriteString("\n")
    }
    
    sql.WriteString(");")
    return sql.String()
}

func (erm *ERModel) PrintModel() {
    fmt.Println("ER Model:")
    
    fmt.Println("\nEntities:")
    for _, entity := range erm.Entities {
        fmt.Printf("  %s:\n", entity.Name)
        fmt.Printf("    Primary Key: %v\n", entity.PrimaryKey)
        fmt.Printf("    Attributes:\n")
        for _, attr := range entity.Attributes {
            fmt.Printf("      %s (%s) - Key: %t\n", attr.Name, attr.DataType, attr.IsKey)
        }
    }
    
    fmt.Println("\nRelationships:")
    for _, rel := range erm.Relationships {
        fmt.Printf("  %s:\n", rel.Name)
        fmt.Printf("    Type: %v\n", rel.Type)
        fmt.Printf("    Entities: %v\n", rel.Entities)
        fmt.Printf("    Cardinality: %v\n", rel.Cardinality)
        if len(rel.Attributes) > 0 {
            fmt.Printf("    Attributes:\n")
            for _, attr := range rel.Attributes {
                fmt.Printf("      %s (%s)\n", attr.Name, attr.DataType)
            }
        }
    }
}

func main() {
    erm := NewERModel()
    
    fmt.Println("ER Model Demo:")
    
    // Create entities
    studentAttributes := []Attribute{
        {Name: "StudentID", Type: Simple, DataType: "INT", IsKey: true},
        {Name: "Name", Type: Simple, DataType: "VARCHAR(100)", IsKey: false},
        {Name: "Email", Type: Simple, DataType: "VARCHAR(100)", IsKey: false},
        {Name: "Phone", Type: MultiValued, DataType: "VARCHAR(20)", IsKey: false},
    }
    
    courseAttributes := []Attribute{
        {Name: "CourseID", Type: Simple, DataType: "INT", IsKey: true},
        {Name: "Title", Type: Simple, DataType: "VARCHAR(100)", IsKey: false},
        {Name: "Credits", Type: Simple, DataType: "INT", IsKey: false},
    }
    
    instructorAttributes := []Attribute{
        {Name: "InstructorID", Type: Simple, DataType: "INT", IsKey: true},
        {Name: "Name", Type: Simple, DataType: "VARCHAR(100)", IsKey: false},
        {Name: "Department", Type: Simple, DataType: "VARCHAR(50)", IsKey: false},
    }
    
    erm.AddEntity("Student", studentAttributes)
    erm.AddEntity("Course", courseAttributes)
    erm.AddEntity("Instructor", instructorAttributes)
    
    // Create relationships
    enrollmentAttributes := []Attribute{
        {Name: "Grade", Type: Simple, DataType: "CHAR(2)", IsKey: false},
        {Name: "Semester", Type: Simple, DataType: "VARCHAR(20)", IsKey: false},
    }
    
    erm.AddRelationship("Enrollment", ManyToMany, []string{"Student", "Course"}, enrollmentAttributes)
    erm.AddRelationship("Teaches", OneToMany, []string{"Instructor", "Course"}, []Attribute{})
    
    // Set cardinalities
    erm.SetCardinality("Enrollment", "Student", "Many")
    erm.SetCardinality("Enrollment", "Course", "Many")
    erm.SetCardinality("Teaches", "Instructor", "One")
    erm.SetCardinality("Teaches", "Course", "Many")
    
    // Print model
    erm.PrintModel()
    
    // Generate SQL
    fmt.Println("\nGenerated SQL:")
    sqlStatements := erm.GenerateSQL()
    for _, sql := range sqlStatements {
        fmt.Println(sql)
        fmt.Println()
    }
}
```

## Indexing Strategies

### Theory

Indexing is a database optimization technique that improves query performance by creating data structures that allow faster data retrieval. Common indexing strategies include B-trees, hash indexes, and bitmap indexes.

### Index Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
    "strings"
)

type IndexType int

const (
    BTreeIndex IndexType = iota
    HashIndex
    BitmapIndex
)

type Index struct {
    Name      string
    TableName string
    Column    string
    Type      IndexType
    Entries   []IndexEntry
}

type IndexEntry struct {
    Key   interface{}
    RowID int
}

type DatabaseIndex struct {
    indexes map[string]*Index
}

func NewDatabaseIndex() *DatabaseIndex {
    return &DatabaseIndex{
        indexes: make(map[string]*Index),
    }
}

func (di *DatabaseIndex) CreateIndex(name, tableName, column string, indexType IndexType) *Index {
    index := &Index{
        Name:      name,
        TableName: tableName,
        Column:    column,
        Type:      indexType,
        Entries:   make([]IndexEntry, 0),
    }
    
    di.indexes[name] = index
    fmt.Printf("Created %v index: %s on %s.%s\n", indexType, name, tableName, column)
    return index
}

func (di *DatabaseIndex) InsertEntry(indexName string, key interface{}, rowID int) {
    if index, exists := di.indexes[indexName]; exists {
        entry := IndexEntry{
            Key:   key,
            RowID: rowID,
        }
        
        index.Entries = append(index.Entries, entry)
        
        // Sort entries for B-tree index
        if index.Type == BTreeIndex {
            sort.Slice(index.Entries, func(i, j int) bool {
                return di.compareKeys(index.Entries[i].Key, index.Entries[j].Key)
            })
        }
        
        fmt.Printf("Inserted entry into index %s: key=%v, rowID=%d\n", indexName, key, rowID)
    }
}

func (di *DatabaseIndex) Search(indexName string, key interface{}) []int {
    if index, exists := di.indexes[indexName]; exists {
        switch index.Type {
        case BTreeIndex:
            return di.binarySearch(index, key)
        case HashIndex:
            return di.hashSearch(index, key)
        case BitmapIndex:
            return di.bitmapSearch(index, key)
        }
    }
    
    return []int{}
}

func (di *DatabaseIndex) binarySearch(index *Index, key interface{}) []int {
    var result []int
    
    // Binary search for the key
    left, right := 0, len(index.Entries)-1
    
    for left <= right {
        mid := (left + right) / 2
        cmp := di.compareKeys(key, index.Entries[mid].Key)
        
        if cmp == 0 {
            // Found the key, collect all matching entries
            result = append(result, index.Entries[mid].RowID)
            
            // Check for duplicates to the left
            for i := mid - 1; i >= 0 && di.compareKeys(key, index.Entries[i].Key) == 0; i-- {
                result = append(result, index.Entries[i].RowID)
            }
            
            // Check for duplicates to the right
            for i := mid + 1; i < len(index.Entries) && di.compareKeys(key, index.Entries[i].Key) == 0; i++ {
                result = append(result, index.Entries[i].RowID)
            }
            
            break
        } else if cmp < 0 {
            right = mid - 1
        } else {
            left = mid + 1
        }
    }
    
    return result
}

func (di *DatabaseIndex) hashSearch(index *Index, key interface{}) []int {
    var result []int
    
    // Simple hash search (in real implementation, use proper hash table)
    for _, entry := range index.Entries {
        if di.compareKeys(key, entry.Key) == 0 {
            result = append(result, entry.RowID)
        }
    }
    
    return result
}

func (di *DatabaseIndex) bitmapSearch(index *Index, key interface{}) []int {
    var result []int
    
    // Bitmap search (simplified implementation)
    for _, entry := range index.Entries {
        if di.compareKeys(key, entry.Key) == 0 {
            result = append(result, entry.RowID)
        }
    }
    
    return result
}

func (di *DatabaseIndex) compareKeys(key1, key2 interface{}) int {
    switch k1 := key1.(type) {
    case string:
        k2 := key2.(string)
        return strings.Compare(k1, k2)
    case int:
        k2 := key2.(int)
        if k1 < k2 {
            return -1
        } else if k1 > k2 {
            return 1
        }
        return 0
    default:
        return 0
    }
}

func (di *DatabaseIndex) RangeSearch(indexName string, startKey, endKey interface{}) []int {
    if index, exists := di.indexes[indexName]; exists {
        if index.Type != BTreeIndex {
            fmt.Printf("Range search only supported for B-tree indexes\n")
            return []int{}
        }
        
        var result []int
        
        for _, entry := range index.Entries {
            if di.compareKeys(entry.Key, startKey) >= 0 && di.compareKeys(entry.Key, endKey) <= 0 {
                result = append(result, entry.RowID)
            }
        }
        
        return result
    }
    
    return []int{}
}

func (di *DatabaseIndex) DeleteEntry(indexName string, key interface{}, rowID int) {
    if index, exists := di.indexes[indexName]; exists {
        for i, entry := range index.Entries {
            if di.compareKeys(entry.Key, key) == 0 && entry.RowID == rowID {
                index.Entries = append(index.Entries[:i], index.Entries[i+1:]...)
                fmt.Printf("Deleted entry from index %s: key=%v, rowID=%d\n", indexName, key, rowID)
                break
            }
        }
    }
}

func (di *DatabaseIndex) GetIndexStats(indexName string) {
    if index, exists := di.indexes[indexName]; exists {
        fmt.Printf("Index %s Statistics:\n", indexName)
        fmt.Printf("  Type: %v\n", index.Type)
        fmt.Printf("  Table: %s\n", index.TableName)
        fmt.Printf("  Column: %s\n", index.Column)
        fmt.Printf("  Entries: %d\n", len(index.Entries))
        
        if len(index.Entries) > 0 {
            fmt.Printf("  First Key: %v\n", index.Entries[0].Key)
            fmt.Printf("  Last Key: %v\n", index.Entries[len(index.Entries)-1].Key)
        }
    }
}

func main() {
    di := NewDatabaseIndex()
    
    fmt.Println("Database Index Demo:")
    
    // Create indexes
    di.CreateIndex("idx_student_id", "Student", "StudentID", BTreeIndex)
    di.CreateIndex("idx_student_name", "Student", "Name", BTreeIndex)
    di.CreateIndex("idx_course_id", "Course", "CourseID", HashIndex)
    
    // Insert some data
    di.InsertEntry("idx_student_id", 1, 1)
    di.InsertEntry("idx_student_id", 2, 2)
    di.InsertEntry("idx_student_id", 3, 3)
    di.InsertEntry("idx_student_id", 4, 4)
    di.InsertEntry("idx_student_id", 5, 5)
    
    di.InsertEntry("idx_student_name", "Alice", 1)
    di.InsertEntry("idx_student_name", "Bob", 2)
    di.InsertEntry("idx_student_name", "Charlie", 3)
    di.InsertEntry("idx_student_name", "David", 4)
    di.InsertEntry("idx_student_name", "Eve", 5)
    
    di.InsertEntry("idx_course_id", 101, 1)
    di.InsertEntry("idx_course_id", 102, 2)
    di.InsertEntry("idx_course_id", 103, 3)
    
    // Search operations
    fmt.Println("\nSearch Operations:")
    
    // Search by student ID
    result := di.Search("idx_student_id", 3)
    fmt.Printf("Search for student ID 3: %v\n", result)
    
    // Search by student name
    result = di.Search("idx_student_name", "Bob")
    fmt.Printf("Search for student name 'Bob': %v\n", result)
    
    // Range search
    result = di.RangeSearch("idx_student_id", 2, 4)
    fmt.Printf("Range search for student ID 2-4: %v\n", result)
    
    // Get index statistics
    fmt.Println("\nIndex Statistics:")
    di.GetIndexStats("idx_student_id")
    di.GetIndexStats("idx_student_name")
    di.GetIndexStats("idx_course_id")
    
    // Delete an entry
    fmt.Println("\nDelete Operations:")
    di.DeleteEntry("idx_student_id", 3, 3)
    result = di.Search("idx_student_id", 3)
    fmt.Printf("Search for student ID 3 after deletion: %v\n", result)
}
```

## Follow-up Questions

### 1. Database Normalization
**Q: What are the trade-offs between normalization and denormalization?**
A: Normalization reduces redundancy and improves data integrity but can increase query complexity and join operations. Denormalization improves query performance but increases storage requirements and can lead to data inconsistency.

### 2. Indexing Strategies
**Q: When would you choose a B-tree index over a hash index?**
A: Use B-tree indexes for range queries, sorting, and when you need to support multiple data types. Use hash indexes for exact match queries when you don't need range operations and have a fixed key size.

### 3. Query Optimization
**Q: What factors should you consider when designing database indexes?**
A: Consider query patterns, data distribution, update frequency, storage requirements, and the balance between read and write performance. Also consider the cardinality of indexed columns and the selectivity of queries.

## Complexity Analysis

| Operation | B-Tree Index | Hash Index | Bitmap Index |
|-----------|--------------|------------|--------------|
| Search | O(log n) | O(1) | O(1) |
| Insert | O(log n) | O(1) | O(1) |
| Delete | O(log n) | O(1) | O(1) |
| Range Search | O(log n + k) | O(n) | O(1) |

## Applications

1. **Database Normalization**: Database design, data modeling
2. **ER Modeling**: Database design, system analysis
3. **Indexing**: Query optimization, database performance
4. **Query Optimization**: Database systems, data warehouses

---

**Next**: [Query Optimization](./query-optimization.md) | **Previous**: [Database Systems](../README.md) | **Up**: [Database Systems](../README.md)
